
from dataclasses import dataclass
from queue import Queue, Empty, Full
from datetime import datetime, timezone
import threading
import time
import uuid
import cv2
import numpy as np


@dataclass
class Frame:
    camera_id: str
    frame_id: int
    image: np.ndarray
    utc_ts: float
    mono_ts: float


@dataclass
class StreamLost:
    camera_id: str


def to_iso(timestamp):
    if timestamp is None:
        return None
    return datetime.fromtimestamp(
        timestamp, timezone.utc
    ).isoformat()


# Bounded, low-latency frame queue
class FrameQueue:
    def __init__(self, max_size=4):
        self.queue = Queue(maxsize=max_size)

    def push_drop_oldest(self, item):
        while True:
            try:
                self.queue.put_nowait(item)
                return
            except Full:
                try:
                    self.queue.get_nowait()
                except Empty:
                    pass

    def pop(self):
        return self.queue.get(timeout=0.5)


class CameraReader:
    def __init__(self, camera_id, rtsp_url, frame_queue):
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.frame_queue = frame_queue
        self.running = True
        self.frame_id = 0

    def run(self):
        while self.running:
            cap = cv2.VideoCapture(self.rtsp_url)

            if not cap.isOpened():
                time.sleep(1)
                continue

            while self.running:
                success, image = cap.read()

                if not success:
                    self.frame_queue.push_drop_oldest(
                        StreamLost(self.camera_id)
                    )
                    break

                self.frame_id += 1

                frame = Frame(
                    camera_id=self.camera_id,
                    frame_id=self.frame_id,
                    image=image,
                    utc_ts=time.time(),
                    mono_ts=time.monotonic()
                )

                self.frame_queue.push_drop_oldest(frame)

            cap.release()
            time.sleep(1)



class MotionDetector:
    def __init__(
        self,
        roi,
        pixel_threshold=20,
        motion_ratio=0.012
    ):
        self.roi = roi
        self.pixel_threshold = pixel_threshold
        self.motion_ratio = motion_ratio
        self.previous = None

    def reset(self):
        self.previous = None

    def update(self, frame):
        x1, y1, x2, y2 = self.roi

        # Monitor only the dice/jar region
        crop = frame[y1:y2, x1:x2]

        gray = cv2.cvtColor(
            crop, cv2.COLOR_BGR2GRAY
        )
        gray = cv2.GaussianBlur(
            gray, (5, 5), 0
        )

        if self.previous is None:
            self.previous = gray
            return None

        diff = cv2.absdiff(self.previous, gray)

        _, binary = cv2.threshold(
            diff,
            self.pixel_threshold,
            255,
            cv2.THRESH_BINARY
        )

        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(
            binary, cv2.MORPH_OPEN, kernel
        )

        changed_ratio = (
            np.count_nonzero(binary) / binary.size
        )

        self.previous = gray

        return changed_ratio > self.motion_ratio


class DiceDetector:
    def __init__(self, runtime):
        self.runtime = runtime

    def preprocess(self, frame):
        # Resize, letterbox, normalize, NCHW
        return detector_preprocess(frame)

    def infer(self, tensor):
        return self.runtime.run(tensor)

    def postprocess(self, raw_output, transform):
        # Decode YOLO boxes, reverse letterbox,
        # confidence filtering and NMS.
        return detector_postprocess(
            raw_output, transform
        )

    def detect(self, frame):
        tensor, transform = self.preprocess(frame)

        raw_output = self.infer(tensor)

        boxes = self.postprocess(
            raw_output, transform
        )

        # Keep the ordering deterministic
        boxes.sort(
            key=lambda b: (b.x1 + b.x2) / 2
        )

        return boxes


class DiceClassifier:
    def __init__(self, runtime):
        self.runtime = runtime

    def preprocess(self, crops):
        # Resize each dice crop and create a batch
        return classifier_preprocess(crops)

    def infer(self, batch):
        return self.runtime.run(batch)

    def postprocess(self, logits):
        probabilities = softmax(logits, axis=1)

        results = []
        for row in probabilities:
            results.append({
                "value": int(np.argmax(row)) + 1,
                "confidence": float(np.max(row))
            })

        return results

    def predict(self, frame, boxes):
        crops = []

        for box in boxes:
            crop = frame[
                box.y1:box.y2,
                box.x1:box.x2
            ]
            crops.append(crop)

        batch = self.preprocess(crops)
        logits = self.infer(batch)

        return self.postprocess(logits)



class GameProcessor:
    def __init__(self, motion, detector, classifier, publisher):
        self.motion = motion
        self.detector = detector
        self.classifier = classifier
        self.publisher = publisher

        self.state = "IDLE"
        self.start_time = None
        self.stop_time = None

    def process_frame(self, frame, timestamp):
        moving = self.motion.detect(frame)

        if self.state == "IDLE":
            if moving:
                self.start_time = timestamp
                self.state = "ROLLING"

        elif self.state == "ROLLING":
            if not moving:
                self.stop_time = timestamp
                self.state = "SETTLING"

        elif self.state == "SETTLING":
            if moving:
                self.state = "ROLLING"

            elif timestamp - self.stop_time >= 0.4:
                boxes = self.detector.detect(frame)

                if len(boxes) != 3:
                    return  # Retry on another stable frame

                values = self.classifier.predict(frame, boxes)

                if not valid(values):
                    return  # Retry or report incomplete

                result = {
                    "start_time": self.start_time,
                    "end_time": self.stop_time,
                    "dice_values": values
                }

                self.publisher.publish(result)

                self.state = "IDLE"
                self.start_time = None
                self.stop_time = None




# Interfaces below are pseudocode adapters.
# Outbox, broker, database and websocket_hub
# are implemented using chosen infrastructure.


class ResultPublisher:
    def __init__(self, outbox):
        self.outbox = outbox

    def publish(self, event):
        # Durable local write, idempotent by event_id.
        # Returns only after local persistence.
        self.outbox.insert_if_absent(
            key=event["event_id"],
            value=event
        )


class OutboxRelay:
    def __init__(self, outbox, broker):
        self.outbox = outbox
        self.broker = broker

    def run(self):
        while True:
            for record in self.outbox.unsent():
                try:
                    self.broker.publish(
                        topic="dice.game.results",
                        key=record["game_id"],
                        payload=record
                    )
                    self.outbox.mark_sent(
                        record["event_id"]
                    )
                except Exception:
                    # Retry on the next iteration
                    pass

            time.sleep(0.1)


class DBConsumer:
    def __init__(self, broker, database):
        self.broker = broker
        self.database = database

    def run(self):
        for message in self.broker.consume(
            topic="dice.game.results",
            group="dice-db"
        ):
            event = message.payload

            try:
                # DB operation must be idempotent.
                # game_id has a UNIQUE constraint.
                self.database.insert_if_absent(
                    table="games",
                    unique_key="game_id",
                    data=event
                )

                # Notification only after DB commit
                self.broker.publish(
                    topic="dice.game.saved",
                    key=event["game_id"],
                    payload=event
                )

                message.ack()

            except Exception:
                message.retry()


class ClientNotifier:
    def __init__(self, broker, websocket_hub):
        self.broker = broker
        self.websocket_hub = websocket_hub

    def run(self):
        for message in self.broker.consume(
            topic="dice.game.saved",
            group="dice-notifier"
        ):
            event = message.payload

            try:
                self.websocket_hub.broadcast(
                    camera_id=event["camera_id"],
                    data=event
                )

                message.ack()

            except Exception:
                message.retry()


#===========================


def start_thread(target):
    thread = threading.Thread(
        target=target,
        daemon=True
    )
    thread.start()
    return thread


def start_system(camera_configs):
    # Infrastructure adapters
    broker = create_message_broker()
    database = create_database_connection()
    websocket_hub = create_websocket_hub()
    outbox = create_durable_outbox()

    publisher = ResultPublisher(outbox)

    # Background infrastructure workers
    relay = OutboxRelay(outbox, broker)
    db_consumer = DBConsumer(broker, database)
    notifier = ClientNotifier(
        broker, websocket_hub
    )

    start_thread(relay.run)
    start_thread(db_consumer.run)
    start_thread(notifier.run)

    # Start one logical pipeline per camera
    for config in camera_configs:
        camera_id = config["camera_id"]

        frame_queue = FrameQueue(max_size=4)

        reader = CameraReader(
            camera_id=camera_id,
            rtsp_url=config["rtsp_url"],
            frame_queue=frame_queue
        )

        motion = MotionDetector(
            roi=config["jar_roi"]
        )

        detector = DiceDetector(
            runtime=load_dice_detector()
        )

        classifier = DiceClassifier(
            runtime=load_dice_classifier()
        )

        processor = GameProcessor(
            camera_id=camera_id,
            frame_queue=frame_queue,
            motion_detector=motion,
            dice_detector=detector,
            dice_classifier=classifier,
            publisher=publisher
        )

        start_thread(reader.run)
        start_thread(processor.run)


camera_configs = [
    {
        "camera_id": "camera_01",
        "rtsp_url": "rtsp://camera-01/stream",
        "jar_roi": (100, 100, 500, 500)
    }
]

start_system(camera_configs)
