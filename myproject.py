from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import cv2
import pyttsx3

Builder.load_file('design.kv')

class ObjectDetectionApp(App):
    def build(self):
        self.text_speech = pyttsx3.init()

        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 640)
        self.cap.set(4, 480)

        self.classNames = []
        classFile = "coco.names"
        with open(classFile, 'rt') as f:
            self.classNames = f.read().rstrip('\n').split('\n')

        configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
        weightsPath = 'frozen_inference_graph.pb'

        self.net = cv2.dnn_DetectionModel(weightsPath, configPath)
        self.net.setInputSize(320, 320)
        self.net.setInputScale(1.0 / 127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)

        # Calibration factor (to be adjusted based on your measurements)
        self.calibration_factor = 1  # Adjust this based on your specific setup

        self.layout = BoxLayout(orientation='vertical')

        self.image_widget = Image()
        self.layout.add_widget(self.image_widget)

        # Update the camera feed and object detection every 1/30 seconds (30 FPS)
        Clock.schedule_interval(self.update, 1.0 / 30.0)

        return self.layout

    def update(self, dt):
        success, img = self.cap.read()
        classIds, confs, bbox = self.net.detect(img, confThreshold=0.5)

        if len(self.classNames) > 0 and len(classIds) > 0:
            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                x, y, w, h = box
                x_center = x + w / 2
                y_center = y + h / 2
                width, height = img.shape[1], img.shape[0]

                # Add object location description
                if x_center < width / 3:
                    location_info = "left side"
                elif x_center > 2 * width / 3:
                    location_info = "right side"
                else:
                    location_info = "center"

                if y_center < height / 3:
                    location_info += " upper side"
                elif y_center > 2 * height / 3:
                    location_info += " lower side"
                else:
                    location_info += " middle"

                # Depth estimation
                mid_x = (x_center + x_center) / 2
                mid_y = (y_center + y_center) / 2
                apx_distance = round(((1 - (w / width)) ** 4) * self.calibration_factor, 2)  # Convert to meters

                # Check conditions for depth estimation
                if apx_distance < 5:  # Adjust this threshold based on your measurements
                    depth_info = "near"
                else:
                    depth_info = "far"

                # Check if classId is within the valid range
                if 0 <= classId - 1 < len(self.classNames):
                    # Add object distance description
                    description = f"{self.classNames[classId - 1].upper()} in {location_info} at a {depth_info} distance of {apx_distance:.2f} meters"

                    # Draw rectangle and display description
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(img, description, (x + 10, y + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255.0), 2)

                    # Print and speak the description
                    print(description)
                    self.text_speech.say(description)
                    self.text_speech.runAndWait()

        # Display the camera feed in the Kivy Image widget
        buf1 = cv2.flip(img, 0)
        buf = buf1.tostring()
        texture = Texture.create(size=(img.shape[1], img.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.image_widget.texture = texture

    def on_stop(self):
        # Release the camera when the app is closed
        self.cap.release()

if __name__ == '__main__':
    ObjectDetectionApp().run()

