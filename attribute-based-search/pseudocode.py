"""
PERSON ATTRIBUTE-BASED IMAGE SEARCH | SINGLE-FILE LLD PSEUDOCODE
================================================================
Scope: 200 RTSP cameras; sampled inference; person tracking; attributes;
      60-day OpenSearch indexing, crop storage, and authorized search.

This is IMPLEMENTATION-LEVEL PSEUDOCODE written with Python syntax.
External APIs (RTSP/YOLO/ByteTrack/Triton/Kafka/OpenSearch/MinIO) are adapters;
replace `...` / SDK_... placeholders with your chosen SDK calls.

Architectural invariants:
  1. Exactly one active tracking session holds the lease for each camera.
  2. Detection batches may mix cameras; tracker input is ordered per camera.
  3. One observation ID per local person track; publish revisions (not duplicates).
  4. Crops are stored before observation events are advertised to the indexer.
  5. Outbox + at-least-once Kafka + version-aware upsert make retries safe.
  6. Every search applies camera-level RBAC and rolling 60-day time constraints.
  7. Unknown attributes are retained as UNKNOWN, never invented.
"""

from __future__ import annotations

import asyncio
import math
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Optional


# -----------------------------------------------------------------------------
# 1) DATA MODEL / CONTRACTS
# -----------------------------------------------------------------------------

UTC = timezone.utc
RETENTION_DAYS = 60
ANALYTICS_FPS = 5.0
MAX_FRAME_QUEUE = 32
MAX_BEST_CROPS = 3
INITIAL_PUBLISH_AFTER_S = 2.0
UPDATE_INTERVAL_S = 5.0


def utc_now() -> datetime:
    return datetime.now(UTC)


class TrackStatus(str, Enum):
    ACTIVE = "active"
    FINALIZED = "finalized"


@dataclass(frozen=True)
class CameraConfig:
    camera_id: str
    rtsp_url: str
    analytics_fps: float = ANALYTICS_FPS
    enabled: bool = True
    zone_id: str = "default"

    def validate(self) -> None:
        assert self.camera_id and self.rtsp_url
        assert 0 < self.analytics_fps <= 30


@dataclass(frozen=True)
class LeaseToken:
    camera_id: str
    session_id: str
    worker_id: str
    fencing_token: int  # increases on every new owner; rejects stale workers
    expires_at: datetime


@dataclass
class FramePacket:
    camera_id: str
    session_id: str
    frame_seq: int
    captured_at: datetime
    image: Any   # decoded image or local frame reference (not Kafka payload)


@dataclass
class Detection:
    bbox_xyxy: tuple[int, int, int, int]
    score: float
    class_name: str = "person"


@dataclass
class DetectionResult:
    frame: FramePacket
    persons: list[Detection]


@dataclass
class PersonCrop:
    image: Any
    bbox_xyxy: tuple[int, int, int, int]
    timestamp: datetime
    quality: float


@dataclass
class AttributeScores:
    # Map attribute -> map label -> calibrated probability or model score.
    distributions: dict[str, dict[str, float]] = field(default_factory=dict)
    visibility: dict[str, bool] = field(default_factory=dict)
    model_version: str = "attrs-v1"

    def highest(self, attribute: str, min_score: float = 0.0) -> tuple[str, float]:
        if not self.visibility.get(attribute, False):
            return "unknown", 0.0
        probs = self.distributions.get(attribute, {})
        if not probs:
            return "unknown", 0.0
        label = max(probs, key=probs.get)
        score = float(probs[label])
        return (label, score) if score >= min_score else ("unknown", score)


@dataclass
class TrackState:
    observation_id: str
    camera_id: str
    session_id: str
    track_id: int
    first_seen: datetime
    last_seen: datetime
    best_crops: list[PersonCrop] = field(default_factory=list)
    version: int = 0
    status: TrackStatus = TrackStatus.ACTIVE
    last_published_at: Optional[datetime] = None

    def update(self, timestamp: datetime) -> None:
        self.last_seen = max(self.last_seen, timestamp)

    def should_publish(self, now: datetime) -> bool:
        age_s = (self.last_seen - self.first_seen).total_seconds()
        if age_s < INITIAL_PUBLISH_AFTER_S or not self.best_crops:
            return False
        if self.last_published_at is None:
            return True
        return (now - self.last_published_at).total_seconds() >= UPDATE_INTERVAL_S

    def next_version(self, now: datetime) -> int:
        self.version += 1
        self.last_published_at = now
        return self.version


@dataclass
class PersonObservation:
    observation_id: str
    camera_id: str
    first_seen: datetime
    last_seen: datetime
    attributes: AttributeScores
    crop_key: str
    version: int
    status: TrackStatus
    model_version: str

    def index_name(self) -> str:
        # The observation always stays in the same daily index, even if updated.
        return f"person-obs-{self.first_seen:%Y.%m.%d}"

    def to_document(self) -> dict[str, Any]:
        labels = {}
        confidences = {}
        for attr in self.attributes.distributions:
            labels[attr], confidences[attr] = self.attributes.highest(attr)
        return {
            "observation_id": self.observation_id,
            "camera_id": self.camera_id,
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "labels": labels,
            "confidence": confidences,
            "distributions": self.attributes.distributions,
            "visibility": self.attributes.visibility,
            "crop_key": self.crop_key,
            "version": self.version,
            "status": self.status.value,
            "model_version": self.model_version,
        }


@dataclass
class SearchQuery:
    start_time: datetime
    end_time: datetime
    camera_ids: set[str]
    filters: dict[str, str]  # e.g. {shirt_color:red, pants_color:black}
    age_range: Optional[tuple[int, int]] = None  # approximate; mapped to groups
    min_confidence: float = 0.50
    strict: bool = True
    limit: int = 50
    cursor: Optional[str] = None

    def validate(self) -> None:
        assert self.start_time <= self.end_time
        assert 1 <= self.limit <= 100
        assert 0 <= self.min_confidence <= 1
        if self.age_range is not None:
            assert 0 <= self.age_range[0] <= self.age_range[1] <= 120


@dataclass
class SearchHit:
    observation_id: str
    camera_id: str
    timestamp: datetime
    attributes: dict[str, str]
    confidence: dict[str, float]
    rank_score: float
    image_url: str


# -----------------------------------------------------------------------------
# 2) CAMERA MANAGEMENT / INGESTION
# -----------------------------------------------------------------------------

class CameraRegistry:
    def __init__(self, config_store: Any):
        self.config_store = config_store

    async def list_active(self) -> list[CameraConfig]:
        return await self.config_store.list_cameras(enabled=True)

    async def get_camera(self, camera_id: str) -> CameraConfig:
        return await self.config_store.get_camera(camera_id)

    async def set_health(self, camera_id: str, healthy: bool) -> None:
        await self.config_store.update_health(camera_id, healthy)


class CameraLeaseManager:
    def __init__(self, distributed_lease_store: Any, ttl_seconds: int = 15):
        self.store = distributed_lease_store
        self.ttl_seconds = ttl_seconds

    async def acquire(self, camera_id: str, worker_id: str) -> Optional[LeaseToken]:
        # Atomic CAS in Redis / etcd / PostgreSQL lease table.
        # Return None if an unexpired camera lease already exists.
        return await self.store.acquire_fenced_lease(camera_id, worker_id, self.ttl_seconds)

    async def renew(self, lease: LeaseToken) -> bool:
        return await self.store.renew_if_owner(lease, self.ttl_seconds)

    async def release(self, lease: LeaseToken) -> None:
        await self.store.release_if_owner(lease)

    async def is_owner(self, lease: LeaseToken) -> bool:
        return await self.store.check_fencing_token(lease)


class FrameSampler:
    def __init__(self, target_fps: float):
        self.interval_seconds = 1.0 / target_fps
        self.last_emitted: Optional[datetime] = None

    def accept(self, captured_at: datetime) -> bool:
        if self.last_emitted is None or (
            captured_at - self.last_emitted
        ).total_seconds() >= self.interval_seconds:
            self.last_emitted = captured_at
            return True
        return False


class CameraWorker:
    def __init__(self, camera: CameraConfig, worker_id: str,
                 leases: CameraLeaseManager, frame_queue: Any,
                 registry: CameraRegistry):
        self.camera = camera
        self.worker_id = worker_id
        self.leases = leases
        self.frame_queue = frame_queue
        self.registry = registry
        self.sampler = FrameSampler(camera.analytics_fps)
        self.frame_seq = 0

    async def start(self) -> None:
        lease = await self.leases.acquire(self.camera.camera_id, self.worker_id)
        if lease is None:
            return  # Another worker owns this camera.
        session_id = lease.session_id
        try:
            while await self.leases.is_owner(lease):
                try:
                    # RTSP adapter yields (decoded_frame, UTC_timestamp).
                    async for image, timestamp in sdk_rtsp_frames(self.camera.rtsp_url):
                        if not await self.leases.renew(lease):
                            return
                        if not self.sampler.accept(timestamp):
                            continue
                        self.frame_seq += 1
                        packet = FramePacket(
                            self.camera.camera_id, session_id,
                            self.frame_seq, timestamp, image
                        )
                        # put_latest uses a bounded queue: discard oldest if full.
                        await self.frame_queue.put_latest(packet, key=self.camera.camera_id)
                    await self.registry.set_health(self.camera.camera_id, False)
                except ConnectionError:
                    await self.registry.set_health(self.camera.camera_id, False)
                await asyncio.sleep(1)  # production: exponential backoff + jitter
        finally:
            await self.leases.release(lease)

    async def stop(self) -> None:
        # Production: stop signal cancels RTSP and releases camera lease.
        ...


# -----------------------------------------------------------------------------
# 3) GPU DETECTION / ORDERED TRACKING
# -----------------------------------------------------------------------------

class PersonDetector:
    def __init__(self, inference_engine: Any, threshold: float = 0.35):
        self.engine = inference_engine
        self.threshold = threshold

    async def detect_batch(self, batch: list[FramePacket]) -> list[DetectionResult]:
        raw_predictions = await self.engine.infer([packet.image for packet in batch])
        results = []
        for packet, predictions in zip(batch, raw_predictions):
            persons = [
                Detection(tuple(p.bbox), float(p.score))
                for p in predictions
                if p.class_name == "person" and p.score >= self.threshold
            ]
            results.append(DetectionResult(frame=packet, persons=persons))
        return results


class OrderedDetectionRouter:
    def __init__(self, tracker_registry: Any):
        self.tracker_registry = tracker_registry
        self.next_seq: dict[tuple[str, str], int] = {}
        self.pending: dict[tuple[str, str], dict[int, DetectionResult]] = {}

    async def route_ordered(self, result: DetectionResult) -> None:
        key = (result.frame.camera_id, result.frame.session_id)
        self.pending.setdefault(key, {})[result.frame.frame_seq] = result
        self.next_seq.setdefault(key, 1)
        # Flush frames in camera order even when GPU batches finish out of order.
        while self.next_seq[key] in self.pending[key]:
            ordered = self.pending[key].pop(self.next_seq[key])
            tracker = await self.tracker_registry.get_or_create(*key)
            await tracker.on_detection(ordered)
            self.next_seq[key] += 1
        # Production: detect missing seq numbers and advance with gap policy,
        # otherwise a dropped/failed frame can block this camera forever.


class DetectionBatcher:
    def __init__(self, frame_queue: Any, detector: PersonDetector,
                 router: OrderedDetectionRouter, max_batch: int = 32):
        self.frame_queue = frame_queue
        self.detector = detector
        self.router = router
        self.max_batch = max_batch

    async def run(self) -> None:
        while True:
            batch = await self.frame_queue.take_batch(
                max_items=self.max_batch, timeout_ms=20
            )
            if not batch:
                continue
            predictions = await self.detector.detect_batch(batch)
            for result in predictions:
                await self.router.route_ordered(result)


class PersonTracker:
    def __init__(self, mot_engine: Any):
        self.engine = mot_engine  # e.g. ByteTrack or BoT-SORT

    def update(self, detections: list[Detection], timestamp: datetime) -> Any:
        # Returns active and terminated tracks with track_id and bbox.
        return self.engine.update(detections, timestamp)

    def finish_stale_tracks(self, timestamp: datetime) -> list[Any]:
        return self.engine.pop_expired(timestamp)


class CropSelector:
    def __init__(self, max_crops: int = MAX_BEST_CROPS):
        self.max_crops = max_crops

    def quality_score(self, crop: Any) -> float:
        # Score sharpness, illumination, occlusion, visible garment area,
        # person size; bounded to [0,1].
        return sdk_quality_estimator(crop)

    def add_if_good(self, track: TrackState, crop: PersonCrop) -> None:
        if crop.quality < 0.5:
            return
        track.best_crops.append(crop)
        track.best_crops.sort(key=lambda c: c.quality, reverse=True)
        track.best_crops = track.best_crops[:self.max_crops]


class CameraTracker:
    def __init__(self, camera_id: str, session_id: str,
                 tracker: PersonTracker, selector: CropSelector,
                 attribute_service: AttributeService):
        self.camera_id = camera_id
        self.session_id = session_id
        self.tracker = tracker
        self.selector = selector
        self.attribute_service = attribute_service
        self.states: dict[int, TrackState] = {}

    async def on_detection(self, result: DetectionResult) -> None:
        update = self.tracker.update(result.persons, result.frame.captured_at)
        for visible in update.active_tracks:
            track_id = int(visible.track_id)
            state = self.states.get(track_id)
            if state is None:
                state = TrackState(
                    observation_id=f"{self.camera_id}:{self.session_id}:{track_id}",
                    camera_id=self.camera_id, session_id=self.session_id,
                    track_id=track_id, first_seen=result.frame.captured_at,
                    last_seen=result.frame.captured_at,
                )
                self.states[track_id] = state
            state.update(result.frame.captured_at)
            crop_image = sdk_crop(result.frame.image, visible.bbox)
            score = self.selector.quality_score(crop_image)
            self.selector.add_if_good(
                state,
                PersonCrop(crop_image, tuple(visible.bbox),
                           result.frame.captured_at, score),
            )
            if visible.is_confirmed and state.should_publish(utc_now()):
                # Snapshot is detached from mutable track state.
                version = state.next_version(utc_now())
                await self.attribute_service.enqueue_snapshot(state, version)

        for finished_track in update.finished_tracks:
            await self._finalize(finished_track.track_id)

    async def _finalize(self, track_id: int) -> None:
        state = self.states.pop(track_id, None)
        if state is None or not state.best_crops:
            return
        state.status = TrackStatus.FINALIZED
        version = state.next_version(utc_now())
        await self.attribute_service.enqueue_snapshot(state, version)

    async def on_camera_shutdown(self) -> None:
        for track_id in list(self.states):
            await self._finalize(track_id)


# -----------------------------------------------------------------------------
# 4) ATTRIBUTE CLASSIFICATION, MULTI-FRAME FUSION
# -----------------------------------------------------------------------------

class AttributeModel:
    def __init__(self, model_version: str, engine: Any):
        self.model_version = model_version
        self.engine = engine

    async def predict_batch(self, images: list[Any]) -> list[dict[str, float]]:
        raise NotImplementedError


class AgeGroupModel(AttributeModel):
    async def predict_batch(self, images: list[Any]) -> list[dict[str, float]]:
        return await self.engine.infer_ages(images)


class GenderAppearanceModel(AttributeModel):
    # Classifies visible presentation, NOT a person's verified gender identity.
    async def predict_batch(self, images: list[Any]) -> list[dict[str, float]]:
        return await self.engine.infer_gender_presentation(images)


class ClothingColorModel(AttributeModel):
    async def predict_batch(self, images: list[Any]) -> list[dict[str, float]]:
        return await self.engine.infer_colors(images)


class AttributeService:
    def __init__(self, age_model: AgeGroupModel,
                 gender_model: GenderAppearanceModel,
                 clothing_model: ClothingColorModel,
                 garment_segmenter: Any, publisher: ObservationPublisher,
                 job_queue: Any):
        self.age_model = age_model
        self.gender_model = gender_model
        self.clothing_model = clothing_model
        self.garment_segmenter = garment_segmenter
        self.publisher = publisher
        self.job_queue = job_queue

    async def enqueue_snapshot(self, track: TrackState, version: int) -> None:
        # Deep-copy immutable state: later updates cannot mutate queued jobs.
        snapshot = sdk_clone_track(track)
        snapshot.version = version
        await self.job_queue.put(snapshot)

    async def classify(self, track: TrackState) -> AttributeScores:
        crops = track.best_crops
        images = [c.image for c in crops]
        ages = await self.age_model.predict_batch(images)
        genders = await self.gender_model.predict_batch(images)

        regions = await self.garment_segmenter.segment_batch(images)
        shirt_inputs = [r.upper_garment for r in regions if r.upper_visible]
        pant_inputs = [r.lower_garment for r in regions if r.lower_visible]
        shirt_predictions = await self.clothing_model.predict_batch(shirt_inputs)
        pant_predictions = await self.clothing_model.predict_batch(pant_inputs)

        shirts = sdk_align_predictions(regions, shirt_predictions, zone="upper")
        pants = sdk_align_predictions(regions, pant_predictions, zone="lower")

        return AttributeScores(
            distributions={
                "age_group": weighted_prob_mean(ages, crops),
                "apparent_gender": weighted_prob_mean(genders, crops),
                "shirt_color": weighted_prob_mean(shirts, crops),
                "pants_color": weighted_prob_mean(pants, crops),
            },
            visibility={
                "age_group": bool(crops),
                "apparent_gender": bool(crops),
                "shirt_color": any(r.upper_visible for r in regions),
                "pants_color": any(r.lower_visible for r in regions),
            },
            model_version=self.age_model.model_version,
        )

    async def run(self) -> None:
        while True:
            snapshots = await self.job_queue.take_batch(max_items=16)
            # Production: merge crops across snapshots into GPU microbatches.
            for track in snapshots:
                attributes = await self.classify(track)
                await self.publisher.persist_and_enqueue(track, attributes)


def weighted_prob_mean(predictions: list[Optional[dict[str, float]]],
                       crops: list[PersonCrop]) -> dict[str, float]:
    numerator: dict[str, float] = {}
    denominator = 0.0
    for prediction, crop in zip(predictions, crops):
        if prediction is None:
            continue
        weight = max(crop.quality, 0.01)
        denominator += weight
        for label, probability in prediction.items():
            numerator[label] = numerator.get(label, 0.0) + weight * probability
    if denominator == 0:
        return {}
    return {label: total / denominator for label, total in numerator.items()}


# -----------------------------------------------------------------------------
# 5) PERSISTENCE / OUTBOX / KAFKA / INDEXING
# -----------------------------------------------------------------------------

class CropStore:
    def __init__(self, s3_client: Any, bucket: str):
        self.client = s3_client
        self.bucket = bucket

    async def put(self, key: str, crop: Any) -> str:
        await self.client.put_object(self.bucket, key, sdk_encode_webp(crop))
        return key

    async def signed_url(self, key: str, ttl_s: int = 120) -> str:
        return await self.client.presign_get(self.bucket, key, expires_s=ttl_s)

    async def delete_many(self, keys: list[str]) -> None:
        await self.client.delete_objects(self.bucket, keys)


class OutboxStore:
    def __init__(self, transactional_store: Any):
        self.store = transactional_store

    async def append(self, event: dict[str, Any]) -> None:
        # Unique key (observation_id, version) prevents duplicate enqueue.
        await self.store.insert_if_absent(event, event_id=event["event_id"])

    async def get_unsent(self, limit: int) -> list[dict[str, Any]]:
        return await self.store.fetch_pending(limit)

    async def mark_sent(self, event_id: str) -> None:
        await self.store.set_published(event_id)


class EventBus:
    def __init__(self, kafka_client: Any, topic: str = "person-observations"):
        self.kafka = kafka_client
        self.topic = topic

    async def publish(self, event: dict[str, Any]) -> None:
        # Partition by observation ID to preserve per-track update ordering.
        await self.kafka.send_and_wait(
            self.topic, key=event["observation_id"], value=event
        )

    async def consume_batch(self, limit: int) -> Any:
        return await self.kafka.consume_batch(self.topic, max_records=limit)

    async def ack(self, committed_offsets: Any) -> None:
        await self.kafka.commit(committed_offsets)


class ObservationPublisher:
    def __init__(self, crops: CropStore, outbox: OutboxStore):
        self.crops = crops
        self.outbox = outbox

    async def persist_and_enqueue(self, track: TrackState,
                                  attrs: AttributeScores) -> None:
        # Stable object key; replays for the same version are idempotent.
        key = f"crops/{track.first_seen:%Y/%m/%d}/{track.observation_id}/v{track.version}.webp"
        best = max(track.best_crops, key=lambda c: c.quality)
        await self.crops.put(key, best.image)

        observation = PersonObservation(
            observation_id=track.observation_id,
            camera_id=track.camera_id,
            first_seen=track.first_seen,
            last_seen=track.last_seen,
            attributes=attrs,
            crop_key=key,
            version=track.version,
            status=track.status,
            model_version=attrs.model_version,
        )
        event = {
            "event_id": f"{observation.observation_id}:{observation.version}",
            "observation_id": observation.observation_id,
            "version": observation.version,
            "index_name": observation.index_name(),
            "document": observation.to_document(),
        }
        await self.outbox.append(event)


async def outbox_relay(outbox: OutboxStore, bus: EventBus) -> None:
    while True:
        for event in await outbox.get_unsent(limit=100):
            await bus.publish(event)
            await outbox.mark_sent(event["event_id"])
        await asyncio.sleep(0.1)


class SearchIndex:
    def __init__(self, open_search_client: Any):
        self.client = open_search_client

    async def bulk_versioned_upsert(self, events: list[dict[str, Any]]) -> Any:
        operations = [
            {
                "_index": e["index_name"],
                "_id": e["observation_id"],
                "_version": e["version"],
                "version_type": "external_gte",
                "_source": e["document"],
            }
            for e in events
        ]
        return await self.client.bulk_index(operations)

    async def search(self, dsl: dict[str, Any], limit: int,
                     cursor: Optional[str]) -> Any:
        return await self.client.search_pit(dsl, size=limit, search_after=cursor)

    async def expired_observations(self, cutoff: datetime) -> Any:
        return await self.client.scan_by_last_seen_before(cutoff)

    async def delete_observations(self, ids: list[tuple[str, str]]) -> None:
        await self.client.bulk_delete(index_and_id_pairs=ids)


class ObservationIndexer:
    def __init__(self, bus: EventBus, index: SearchIndex):
        self.bus = bus
        self.index = index

    async def run(self) -> None:
        while True:
            batch = await self.bus.consume_batch(limit=200)
            if not batch:
                continue
            result = await self.index.bulk_versioned_upsert(batch.events)
            # Retry partial failures and advance only contiguous safe offsets.
            await sdk_retry_transient_index_failures(result)
            safe_offsets = sdk_contiguous_successful_offsets(batch, result)
            await self.bus.ack(safe_offsets)


# -----------------------------------------------------------------------------
# 6) AUTHORIZED ATTRIBUTE SEARCH (NO SAME-PERSON RE-ID CLAIM)
# -----------------------------------------------------------------------------

class AccessPolicy:
    def __init__(self, authorization_db: Any):
        self.db = authorization_db

    async def allowed_cameras(self, user_id: str) -> set[str]:
        return set(await self.db.camera_permissions(user_id))

    async def authorize_query(self, user_id: str,
                              query: SearchQuery) -> set[str]:
        allowed = await self.allowed_cameras(user_id)
        chosen = allowed if not query.camera_ids else allowed & query.camera_ids
        if not chosen:
            raise PermissionError("No permitted cameras")
        return chosen


class PersonSearchService:
    def __init__(self, index: SearchIndex, crops: CropStore,
                 policy: AccessPolicy):
        self.index = index
        self.crops = crops
        self.policy = policy

    async def search(self, user_id: str, query: SearchQuery) -> dict[str, Any]:
        query.validate()
        allowed = await self.policy.authorize_query(user_id, query)
        cutoff = utc_now() - timedelta(days=RETENTION_DAYS)
        start = max(query.start_time, cutoff)
        end = min(query.end_time, utc_now())
        if start > end:
            return {"results": [], "next_cursor": None}

        dsl = self._build_query(query, allowed, start, end)
        response = await self.index.search(dsl, limit=query.limit,
                                           cursor=query.cursor)
        results = []
        for doc in response.hits:
            # Defense in depth: recheck source of each result before signing.
            if doc.camera_id not in allowed or doc.last_seen < cutoff:
                continue
            score = self._relevance(doc, query)
            image_url = await self.crops.signed_url(doc.crop_key)
            results.append(SearchHit(
                observation_id=doc.observation_id,
                camera_id=doc.camera_id,
                timestamp=doc.first_seen,
                attributes=doc.labels,
                confidence=doc.confidence,
                rank_score=score,
                image_url=image_url,
            ))
        results.sort(key=lambda r: (-r.rank_score, r.observation_id))
        return {"results": results, "next_cursor": response.next_cursor}

    def _build_query(self, query: SearchQuery, cameras: set[str],
                     start: datetime, end: datetime) -> dict[str, Any]:
        filters = [
            {"terms": {"camera_id": sorted(cameras)}},
            {"range": {"first_seen": {"lte": end.isoformat()}}},
            {"range": {"last_seen": {"gte": start.isoformat()}}},
        ]
        if query.age_range is not None:
            # Broad categories cannot establish someone's exact age.
            # The boundaries below are model-specific design assumptions.
            age_bins = {
                "child": (0, 17), "young_adult": (18, 34),
                "adult": (35, 59), "elderly": (60, 120),
            }
            low, high = query.age_range
            approximate_groups = [
                name for name, (bin_low, bin_high) in age_bins.items()
                if low <= bin_high and bin_low <= high
            ]
            filters.append({"terms": {"labels.age_group": approximate_groups}})

        for attr, value in query.filters.items():
            if query.strict:
                filters += [
                    {"term": {f"labels.{attr}": value}},
                    {"range": {f"confidence.{attr}": {
                        "gte": query.min_confidence
                    }}},
                ]
            else:
                # Relaxed search can retrieve top-k candidate probabilities.
                filters.append({"range": {
                    f"distributions.{attr}.{value}": {"gte": 0.20}
                }})
        return {"query": {"bool": {"filter": filters}}}

    def _relevance(self, doc: Any, query: SearchQuery) -> float:
        if not query.filters:
            return 1.0
        values = [max(float(doc.distributions.get(attr, {}).get(value, 0)), 1e-6)
                  for attr, value in query.filters.items()]
        return math.exp(sum(math.log(x) for x in values) / len(values))


# -----------------------------------------------------------------------------
# 7) RETENTION, RECONCILIATION, RESTART / HEALTH
# -----------------------------------------------------------------------------

class RetentionService:
    def __init__(self, index: SearchIndex, crops: CropStore):
        self.index = index
        self.crops = crops

    async def purge_expired(self) -> None:
        cutoff = utc_now() - timedelta(days=RETENTION_DAYS)
        # Durable deletion jobs avoid partial-failure leaks. Batch by object key.
        async for docs in self.index.expired_observations(cutoff):
            await sdk_enqueue_durable_delete_job(docs)

        async for job in sdk_pending_delete_jobs():
            # Delete all historical crop versions associated with the observation.
            await self.crops.delete_many(job.all_crop_keys)
            await self.index.delete_observations(job.index_id_pairs)
            await sdk_complete_delete_job(job)

    async def reconcile_orphan_crops(self) -> None:
        # Find crops with no active observation or pending outbox reference.
        # Do not remove new objects awaiting Kafka publication.
        await sdk_object_store_orphan_sweep()


class CameraScheduler:
    def __init__(self, registry: CameraRegistry,
                 leases: CameraLeaseManager, worker_fleet: Any):
        self.registry = registry
        self.leases = leases
        self.worker_fleet = worker_fleet

    async def rebalance(self) -> None:
        cameras = await self.registry.list_active()
        workers = await self.worker_fleet.available_workers()
        for camera in cameras:
            worker = min(workers, key=lambda w: w.estimated_load)
            # Scheduler assigns desired ownership; lease manager enforces it.
            await self.worker_fleet.assign(worker.id, camera.camera_id)


# -----------------------------------------------------------------------------
# 8) ORCHESTRATION SKETCH - ONE DEPLOYMENT, SEPARATE SCALABLE PROCESSES
# -----------------------------------------------------------------------------

async def start_system(dependencies: Any) -> None:
    """Wiring sketch; each `run` loop may be deployed in independent processes."""
    registry = CameraRegistry(dependencies.config_store)
    leases = CameraLeaseManager(dependencies.lease_store)
    crops = CropStore(dependencies.s3, bucket="person-crops")
    outbox = OutboxStore(dependencies.postgres)
    bus = EventBus(dependencies.kafka)
    index = SearchIndex(dependencies.opensearch)

    publisher = ObservationPublisher(crops, outbox)
    attrs = AttributeService(
        age_model=AgeGroupModel("age-v3", dependencies.age_engine),
        gender_model=GenderAppearanceModel("presentation-v2", dependencies.gender_engine),
        clothing_model=ClothingColorModel("color-v4", dependencies.color_engine),
        garment_segmenter=dependencies.garment_segmenter,
        publisher=publisher,
        job_queue=dependencies.attribute_queue,
    )
    router = OrderedDetectionRouter(dependencies.tracker_registry)
    detector = PersonDetector(dependencies.yolo_engine)
    batcher = DetectionBatcher(dependencies.frame_queue, detector, router)
    indexer = ObservationIndexer(bus, index)
    retention = RetentionService(index, crops)
    search_service = PersonSearchService(
        index, crops, AccessPolicy(dependencies.authorization_db)
    )
    # REST API calls search_service.search(user_id, SearchQuery(...)).
    # Tracker registry creates CameraTracker per (camera_id, session_id),
    # using PersonTracker, CropSelector and the shared AttributeService.
    # Scheduler spreads 200 camera workers across the GPU processing nodes.
    await asyncio.gather(
        batcher.run(),
        attrs.run(),
        outbox_relay(outbox, bus),
        indexer.run(),
        dependencies.api.serve(search_service),
        dependencies.scheduler.run(registry, leases),
        dependencies.retention_schedule.run(retention.purge_expired),
    )


# -----------------------------------------------------------------------------
# PLACEHOLDER SDK ADAPTERS - IMPLEMENT FOR ACTUAL BACKENDS
# -----------------------------------------------------------------------------
# sdk_rtsp_frames: async RTSP decoder (GStreamer/FFmpeg)
# sdk_quality_estimator, sdk_crop, sdk_clone_track: image/track utilities
# sdk_align_predictions: restore skipped hidden garments to crop positions
# sdk_encode_webp: compress crop into bytes
# sdk_retry_transient_index_failures, sdk_contiguous_successful_offsets
# sdk_enqueue_durable_delete_job, sdk_pending_delete_jobs,
# sdk_complete_delete_job, sdk_object_store_orphan_sweep
# In real code, these are adapters, not global functions.
