
import cv2
import time
import uuid
import threading

from dataclasses import dataclass
from queue import Queue, Empty, Full


@dataclass
class Frame:
    camera_id: str
    frame_id: int
    image: object
    timestamp: float
    encoder_tick: int
    stream_epoch: int


@dataclass
class InspectionJob:
    product_id: str
    camera_id: str
    image: object
    timestamp: float
    encoder_tick: int


class SharedFrameQueue:
    def __init__(self, max_size=128):
        self.queue = Queue(maxsize=max_size)

    def push(self, frame):
        try:
            self.queue.put(frame, timeout=0.05)
            return True
        except Full:
            return False

    def pop(self, timeout=0.5):
        return self.queue.get(timeout=timeout)


class CameraReader:
    def __init__(
        self, camera_id, rtsp_url,
        frame_queue, encoder, health
    ):
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.frame_queue = frame_queue
        self.encoder = encoder
        self.health = health

        self.frame_id = 0
        self.stream_epoch = 0

    def run(self):
        while True:
            cap = cv2.VideoCapture(self.rtsp_url)

            if not cap.isOpened():
                self.health.camera_down(self.camera_id)
                time.sleep(1)
                continue

            self.stream_epoch += 1

            while True:
                success, image = cap.read()

                if not success:
                    self.health.camera_down(self.camera_id)
                    break

                self.frame_id += 1

                frame = Frame(
                    camera_id=self.camera_id,
                    frame_id=self.frame_id,
                    image=image,
                    timestamp=time.time(),
                    encoder_tick=self.encoder.read(),
                    stream_epoch=self.stream_epoch
                )

                success = self.frame_queue.push(frame)

                if not success:
                    self.health.report_overflow(
                        self.camera_id
                    )
                    self.health.request_safe_handling(
                        self.camera_id
                    )

            cap.release()
            time.sleep(1)




class BatchCollector:
    def __init__(
        self, frame_queue,
        batch_size=8,
        max_wait_ms=10
    ):
        self.queue = frame_queue
        self.batch_size = batch_size
        self.max_wait = max_wait_ms / 1000

    def get_batch(self):
        try:
            first_frame = self.queue.pop(timeout=0.5)
        except Empty:
            return []

        batch = [first_frame]

        deadline = time.monotonic() + self.max_wait

        while len(batch) < self.batch_size:
            remaining = deadline - time.monotonic()

            if remaining <= 0:
                break

            try:
                frame = self.queue.pop(
                    timeout=remaining
                )
                batch.append(frame)

            except Empty:
                break

        return batch



batch = [
    Frame(camera_id="cam1", frame_id=101, ...),
    Frame(camera_id="cam2", frame_id=204, ...),
    Frame(camera_id="cam3", frame_id=93, ...),
    Frame(camera_id="cam4", frame_id=171, ...),
    Frame(camera_id="cam1", frame_id=102, ...),
    Frame(camera_id="cam2", frame_id=205, ...),
    Frame(camera_id="cam3", frame_id=94, ...),
    Frame(camera_id="cam4", frame_id=172, ...)
]




class BatchProductDetector:
    def __init__(self, model):
        self.model = model

    def preprocess_batch(self, images):
        return preprocess_product_images(images)

    def infer_batch(self, tensor):
        return self.model.run(tensor)

    def postprocess_batch(self, outputs):
        return decode_product_boxes(outputs)

    def predict_batch(self, images):
        tensor = self.preprocess_batch(images)

        outputs = self.infer_batch(tensor)

        return self.postprocess_batch(outputs)


class CameraTracker:
    def __init__(self, camera_id, inspection_line):
        self.camera_id = camera_id
        self.inspection_line = inspection_line

        self.tracker = create_object_tracker()
        self.inspected = set()
        self.stream_epoch = None
        self.session_id = uuid.uuid4().hex

    def reset(self, stream_epoch):
        self.tracker = create_object_tracker()
        self.inspected.clear()
        self.session_id = uuid.uuid4().hex
        self.stream_epoch = stream_epoch

    def update(self, detections, frame):
        if self.stream_epoch != frame.stream_epoch:
            self.reset(frame.stream_epoch)

        tracks = self.tracker.update(detections)

        new_products = []

        for track in tracks:
            if track.id in self.inspected:
                continue

            # Detect when the product passes
            # the designated inspection line
            if not track.crossed_line(
                self.inspection_line
            ):
                continue

            product_id = (
                f"{self.camera_id}_"
                f"{self.session_id}_"
                f"{track.id}"
            )

            self.inspected.add(track.id)

            new_products.append({
                "product_id": product_id,
                "bbox": track.bbox
            })

        # Remove expired track state when safe
        self.tracker.cleanup_expired_tracks()

        return new_products




class BatchDefectDetector:
    def __init__(self, model):
        self.model = model

    def preprocess_batch(self, crops):
        return preprocess_defect_images(crops)

    def infer_batch(self, tensor):
        return self.model.run(tensor)

    def postprocess_batch(self, outputs):
        # Each item contains detected defects,
        # confidences and image/model validity.
        return decode_defect_predictions(outputs)

    def predict_batch(self, crops):
        tensor = self.preprocess_batch(crops)

        outputs = self.infer_batch(tensor)

        return self.postprocess_batch(outputs)


class DecisionEngine:
    def __init__(
        self,
        defect_threshold=0.85,
        review_threshold=0.50
    ):
        self.defect_threshold = defect_threshold
        self.review_threshold = review_threshold

    def decide(self, prediction, crop):
        if not check_image_quality(crop):
            return "REVIEW"

        if not prediction["valid"]:
            return "REVIEW"

        defects = prediction["defects"]

        if any(
            d["confidence"] >= self.defect_threshold
            for d in defects
        ):
            return "FAIL"

        if any(
            d["confidence"] >= self.review_threshold
            for d in defects
        ):
            return "REVIEW"

        return "PASS"


predictions = [
    {
        "valid": True,
        "defects": [
            {"type": "scratch", "confidence": 0.94}
        ]
    },
    {
        "valid": True,
        "defects": []
    },
    {
        "valid": True,
        "defects": [
            {"type": "crack", "confidence": 0.97}
        ]
    }
]




class PLCScheduler:
    def __init__(
        self, plc_connections,
        encoders,
        reject_offsets,
        health
    ):
        self.plcs = plc_connections
        self.encoders = encoders
        self.reject_offsets = reject_offsets
        self.health = health

    def schedule_reject(
        self, camera_id, product_id,
        inspection_tick, status
    ):
        # REVIEW handling is defined by plant policy.
        # Here both FAIL and REVIEW are diverted.

        if inspection_tick is None:
            self.health.request_safe_handling(camera_id)
            return "NO_ENCODER_POSITION"

        target_tick = (
            inspection_tick
            + self.reject_offsets[camera_id]
        )

        current_tick = self.encoders[camera_id].read()

        if not enough_lead_time(
            current_tick, target_tick, camera_id
        ):
            self.health.request_safe_handling(camera_id)
            return "MISSED_REJECT_DEADLINE"

        command = {
            "command_id": product_id,
            "product_id": product_id,
            "target_tick": target_tick,
            "action": "REJECT"
        }

        ack = self.plcs[camera_id].schedule(command)

        if not ack:
            self.health.request_safe_handling(camera_id)
            return "PLC_COMMAND_FAILED"

        return "REJECT_SCHEDULED"





class BatchProcessor:
    def __init__(
        self,
        batch_collector,
        product_detector,
        camera_trackers,
        defect_detector,
        decision_engine,
        plc_scheduler,
        result_publisher,
        health
    ):
        self.collector = batch_collector
        self.product_detector = product_detector
        self.trackers = camera_trackers
        self.defect_detector = defect_detector
        self.decision_engine = decision_engine
        self.plc = plc_scheduler
        self.publisher = result_publisher
        self.health = health

    def run(self):
        while True:
            batch = self.collector.get_batch()

            if not batch:
                continue

            self.process_batch(batch)

    def process_batch(self, batch):
        # STEP 1: Extract images
        images = [frame.image for frame in batch]

        # STEP 2: Batch product detection
        try:
            detections = (
                self.product_detector.predict_batch(images)
            )
            assert len(detections) == len(batch)
        except Exception as error:
            for frame in batch:
                self.health.inference_failed(
                    frame.camera_id, str(error)
                )
                self.health.request_safe_handling(
                    frame.camera_id
                )
            return

        inspection_jobs = []

        # STEP 3: Process tracking per camera
        for frame, boxes in zip(batch, detections):
            try:
                tracker = self.trackers[frame.camera_id]

                new_products = tracker.update(
                    detections=boxes,
                    frame=frame
                )

                # STEP 4: Extract new product crops
                for product in new_products:
                    crop = crop_bbox(
                        frame.image,
                        product["bbox"]
                    )

                    job = InspectionJob(
                        product_id=product["product_id"],
                        camera_id=frame.camera_id,
                        image=crop,
                        timestamp=frame.timestamp,
                        encoder_tick=frame.encoder_tick
                    )

                    inspection_jobs.append(job)

            except Exception as error:
                self.health.processing_failed(
                    frame.camera_id, str(error)
                )
                self.health.request_safe_handling(
                    frame.camera_id
                )

        # No new product at inspection line
        if not inspection_jobs:
            return

        # STEP 5: Collect product crops
        crops = [job.image for job in inspection_jobs]

        # STEP 6: Batch defect inference
        try:
            predictions = (
                self.defect_detector.predict_batch(crops)
            )
            assert len(predictions) == len(inspection_jobs)

        except Exception as error:
            self.health.report_model_error(str(error))

            # Failed inspection is not a PASS
            for job in inspection_jobs:
                self.finalize_product(
                    job=job,
                    status="REVIEW",
                    defects=[],
                    reason="defect_inference_failed"
                )
            return

        # STEP 7: Map results to original products
        for job, prediction in zip(
            inspection_jobs, predictions
        ):
            try:
                status = self.decision_engine.decide(
                    prediction,
                    job.image
                )

                self.finalize_product(
                    job=job,
                    status=status,
                    defects=prediction["defects"],
                    reason=None
                )

            except Exception as error:
                self.finalize_product(
                    job=job,
                    status="REVIEW",
                    defects=[],
                    reason=str(error)
                )

    def finalize_product(
        self, job, status, defects, reason
    ):
        # STEP 8: Handle quality decision

        plc_status = "NOT_REQUIRED"

        if status in ("FAIL", "REVIEW"):
            plc_status = self.plc.schedule_reject(
                camera_id=job.camera_id,
                product_id=job.product_id,
                inspection_tick=job.encoder_tick,
                status=status
            )

        # STEP 9: Build JSON
        result = {
            "event_id": f"{job.product_id}:inspection",
            "product_id": job.product_id,
            "camera_id": job.camera_id,
            "timestamp": job.timestamp,
            "status": status,
            "defects": defects,
            "plc_status": plc_status,
            "reason": reason
        }

        # STEP 10: Durable metadata publication
        try:
            self.publisher.publish(result)
        except Exception as error:
            self.health.report_publish_failure(
                job.camera_id, str(error)
            )
            self.health.request_safe_handling(
                job.camera_id
            )




class ResultPublisher:
    def __init__(self, durable_outbox):
        self.outbox = durable_outbox

    def publish(self, result):
        # Write locally and durably before returning
        self.outbox.insert_if_absent(
            key=result["event_id"],
            payload=result
        )


class MetadataRelay:
    def __init__(self, outbox, broker):
        self.outbox = outbox
        self.broker = broker

    def run(self):
        while True:
            events = self.outbox.get_pending(limit=100)

            for event in events:
                try:
                    self.broker.publish_confirmed(
                        topic="inspection.results",
                        key=event["product_id"],
                        payload=event
                    )

                    self.outbox.mark_sent(
                        event["event_id"]
                    )

                except Exception:
                    # Retry later; keep event in outbox
                    break

            time.sleep(0.1)


class DBConsumer:
    def __init__(self, broker, database):
        self.broker = broker
        self.database = database

    def run(self):
        for message in self.broker.consume(
            topic="inspection.results",
            group="inspection-db-workers"
        ):
            event = message.payload

            try:
                # Atomically save result and a
                # dashboard notification event
                with self.database.transaction() as tx:
                    tx.insert_if_absent(
                        table="inspection_results",
                        key=event["event_id"],
                        data=event
                    )

                    tx.insert_if_absent(
                        table="saved_events_outbox",
                        key=event["event_id"],
                        data=event
                    )

                message.ack()

            except Exception:
                message.retry()


class SavedEventRelay:
    def __init__(self, database, broker):
        self.database = database
        self.broker = broker

    def run(self):
        while True:
            events = self.database.pending_saved_events()

            for event in events:
                try:
                    self.broker.publish_confirmed(
                        topic="inspection.saved",
                        key=event["product_id"],
                        payload=event
                    )
                    self.database.mark_event_sent(
                        event["event_id"]
                    )
                except Exception:
                    break

            time.sleep(0.1)


class ClientNotifier:
    def __init__(self, broker, websocket_server):
        self.broker = broker
        self.websocket = websocket_server

    def run(self):
        for message in self.broker.consume(
            topic="inspection.saved",
            group="dashboard-notifier"
        ):
            event = message.payload

            try:
                self.websocket.broadcast(
                    camera_id=event["camera_id"],
                    payload=event
                )

                message.ack()

            except Exception:
                message.retry()



# =========================================================


class SystemManager:
    def __init__(self, config):
        self.config = config
        self.threads = []

    def start_thread(self, target):
        thread = threading.Thread(
            target=target,
            daemon=True
        )
        thread.start()
        self.threads.append(thread)

    def start(self):
        # ----------------------------------
        # 1. Initialize shared infrastructure
        # ----------------------------------
        health = create_health_monitor()
        broker = create_message_broker()
        database = create_database()
        outbox = create_durable_outbox()
        websocket = create_websocket_server()

        frame_queue = SharedFrameQueue(
            max_size=self.config["queue_size"]
        )

        batch_collector = BatchCollector(
            frame_queue=frame_queue,
            batch_size=self.config["batch_size"],
            max_wait_ms=self.config["max_wait_ms"]
        )

        # ----------------------------------
        # 2. Load models only once
        # ----------------------------------
        product_model = load_product_model(
            self.config["product_model_path"]
        )

        defect_model = load_defect_model(
            self.config["defect_model_path"]
        )

        product_detector = BatchProductDetector(
            product_model
        )

        defect_detector = BatchDefectDetector(
            defect_model
        )

        decision_engine = DecisionEngine()

        # ----------------------------------
        # 3. Initialize cameras and trackers
        # ----------------------------------
        camera_trackers = {}
        encoders = {}
        plc_connections = {}
        reject_offsets = {}

        for camera_config in self.config["cameras"]:
            camera_id = camera_config["camera_id"]

            encoders[camera_id] = open_encoder(
                camera_config["encoder_address"]
            )

            plc_connections[camera_id] = connect_plc(
                camera_config["plc_address"]
            )

            reject_offsets[camera_id] = (
                camera_config["reject_offset_ticks"]
            )

            camera_trackers[camera_id] = CameraTracker(
                camera_id=camera_id,
                inspection_line=(
                    camera_config["inspection_line"]
                )
            )

            reader = CameraReader(
                camera_id=camera_id,
                rtsp_url=camera_config["rtsp_url"],
                frame_queue=frame_queue,
                encoder=encoders[camera_id],
                health=health
            )

            # Each camera has its own RTSP thread
            self.start_thread(reader.run)

        # ----------------------------------
        # 4. Initialize PLC scheduler
        # ----------------------------------
        plc_scheduler = PLCScheduler(
            plc_connections=plc_connections,
            encoders=encoders,
            reject_offsets=reject_offsets,
            health=health
        )

        # ----------------------------------
        # 5. Initialize result publisher
        # ----------------------------------
        publisher = ResultPublisher(outbox)

        # ----------------------------------
        # 6. Initialize shared BatchProcessor
        # ----------------------------------
        processor = BatchProcessor(
            batch_collector=batch_collector,
            product_detector=product_detector,
            camera_trackers=camera_trackers,
            defect_detector=defect_detector,
            decision_engine=decision_engine,
            plc_scheduler=plc_scheduler,
            result_publisher=publisher,
            health=health
        )

        # Only ONE batch processing worker
        self.start_thread(processor.run)

        # ----------------------------------
        # 7. Start metadata/backend workers
        # ----------------------------------
        relay = MetadataRelay(outbox, broker)
        consumer = DBConsumer(broker, database)

        saved_relay = SavedEventRelay(
            database, broker
        )

        notifier = ClientNotifier(
            broker, websocket
        )

        self.start_thread(relay.run)
        self.start_thread(consumer.run)
        self.start_thread(saved_relay.run)
        self.start_thread(notifier.run)

        # ----------------------------------
        # 8. Start health monitoring
        # ----------------------------------
        self.start_thread(health.run)

        # Production: add graceful shutdown,
        # resource cleanup and worker supervision.


CONFIG = {
    "queue_size": 128,
    "batch_size": 8,
    "max_wait_ms": 10,

    "product_model_path": "product_detection.onnx",
    "defect_model_path": "defect_detection.onnx",

    "cameras": [
        {
            "camera_id": "cam_01",
            "rtsp_url": "rtsp://camera1/stream",
            "inspection_line": 300,
            "encoder_address": "encoder_01",
            "plc_address": "plc_01",
            "reject_offset_ticks": 2500
        },
        {
            "camera_id": "cam_02",
            "rtsp_url": "rtsp://camera2/stream",
            "inspection_line": 300,
            "encoder_address": "encoder_02",
            "plc_address": "plc_02",
            "reject_offset_ticks": 2500
        },
        {
            "camera_id": "cam_03",
            "rtsp_url": "rtsp://camera3/stream",
            "inspection_line": 300,
            "encoder_address": "encoder_03",
            "plc_address": "plc_03",
            "reject_offset_ticks": 2500
        }
    ]
}


if __name__ == "__main__":
    system = SystemManager(CONFIG)
    system.start()

    while True:
        time.sleep(1)


"""

{
  "event_id": "cam_01_P101_inspection",
  "product_id": "P101",
  "camera_id": "cam_01",
  "timestamp": 1794574800.125,
  "status": "FAIL",
  "defects": [
    {
      "type": "scratch",
      "confidence": 0.94
    }
  ],
  "plc_status": "REJECT_SCHEDULED",
  "reason": null
}
"""
