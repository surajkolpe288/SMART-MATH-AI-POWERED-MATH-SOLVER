import os
import cv2
import PIL
import numpy as np
import google.generativeai as genai
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from mediapipe.python.solutions import hands, drawing_utils
from dotenv import load_dotenv
from warnings import filterwarnings

# Suppress warnings
filterwarnings(action='ignore')


class Calculator:
    def __init__(self):
        # Load the Env File for Secret API Key
        load_dotenv()

        # Initialize a Webcam to Capture Video and Set Width, Height, and Brightness
        self.cap = cv2.VideoCapture(0)
        self.cap.set(propId=cv2.CAP_PROP_FRAME_WIDTH, value=950)
        self.cap.set(propId=cv2.CAP_PROP_FRAME_HEIGHT, value=550)
        self.cap.set(propId=cv2.CAP_PROP_BRIGHTNESS, value=130)

        # Initialize Canvas Image
        self.imgCanvas = np.zeros(shape=(550, 950, 3), dtype=np.uint8)

        # Initialize MediaPipe Hand object
        self.mphands = hands.Hands(max_num_hands=1, min_detection_confidence=0.75)

        # Initialize Variables for Drawing
        self.p1, self.p2 = 0, 0
        self.fingers = []

    def streamlit_config(self):
        # Page configuration
        st.set_page_config(page_title='Calculator', layout="wide")

        # Customize page styling
        page_background_color = """
        <style>
        [data-testid="stHeader"] {background: rgba(0,0,0,0);}
        .block-container {padding-top: 0rem;}
        </style>
        """
        st.markdown(page_background_color, unsafe_allow_html=True)

        # Page title
        st.markdown(f'<h1 style="text-align: center;">Virtual Calculator</h1>', unsafe_allow_html=True)
        add_vertical_space(1)

    def process_frame(self):
        # Reading the Video Capture to return the Success and Image Frame
        success, img = self.cap.read()
        if not success:
            raise RuntimeError("Error: Unable to access the webcam.")

        # Resize the Image
        img = cv2.resize(src=img, dsize=(950, 550))

        # Flip the Image Horizontally for a Later Selfie-View Display
        self.img = cv2.flip(src=img, flipCode=1)

        # Convert BGR to RGB
        self.imgRGB = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

    def process_hands(self):
        # Process the frame for hand landmarks
        result = self.mphands.process(image=self.imgRGB)

        # Initialize landmark list
        self.landmark_list = []

        if result.multi_hand_landmarks:
            for hand_lms in result.multi_hand_landmarks:
                drawing_utils.draw_landmarks(self.img, hand_lms, hands.HAND_CONNECTIONS)

                # Extract landmarks
                for id, lm in enumerate(hand_lms.landmark):
                    h, w, _ = self.img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    self.landmark_list.append([id, cx, cy])

    def identify_fingers(self):
        self.fingers = []
        if self.landmark_list:
            for id in [4, 8, 12, 16, 20]:  # Thumb to Pinky
                if id != 4:
                    self.fingers.append(int(self.landmark_list[id][2] < self.landmark_list[id - 2][2]))
                else:
                    self.fingers.append(int(self.landmark_list[id][1] < self.landmark_list[id - 2][1]))

    def handle_drawing_mode(self):
        if sum(self.fingers) == 2 and self.fingers[0] == self.fingers[1] == 1:  # Drawing mode
            cx, cy = self.landmark_list[8][1], self.landmark_list[8][2]
            if not self.p1 and not self.p2:
                self.p1, self.p2 = cx, cy
            cv2.line(self.imgCanvas, (self.p1, self.p2), (cx, cy), (255, 0, 255), 5)
            self.p1, self.p2 = cx, cy
        elif sum(self.fingers) == 3 and self.fingers[0] == self.fingers[1] == self.fingers[2] == 1:  # Disable drawing
            self.p1, self.p2 = 0, 0
        elif sum(self.fingers) == 2 and self.fingers[0] == self.fingers[2] == 1:  # Erase drawing
            cx, cy = self.landmark_list[12][1], self.landmark_list[12][2]
            cv2.line(self.imgCanvas, (self.p1, self.p2), (cx, cy), (0, 0, 0), 15)
            self.p1, self.p2 = cx, cy
        elif sum(self.fingers) == 2 and self.fingers[0] == self.fingers[4] == 1:  # Reset canvas
            self.imgCanvas = np.zeros((550, 950, 3), np.uint8)

    def blend_canvas_with_feed(self):
        img_gray = cv2.cvtColor(self.imgCanvas, cv2.COLOR_BGR2GRAY)
        _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
        img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
        self.img = cv2.bitwise_or(self.img, self.imgCanvas)

    def analyze_image_with_genai(self):
        imgCanvas = PIL.Image.fromarray(cv2.cvtColor(self.imgCanvas, cv2.COLOR_BGR2RGB))
        genai.configure(api_key=os.getenv("Generative_Language_API_Key"))

        prompt = "Analyze and solve the math equation in the image."
        response = genai.GenerativeModel(model_name="gemini-1.5-flash").generate_content(
            [prompt, imgCanvas]
        )
        return response.text

    def main(self):
        col1, _, col3 = st.columns([0.8, 0.02, 0.18])

        with col1:
            stframe = st.empty()

        with col3:
            st.markdown('<h5 style="color:green;">OUTPUT:</h5>', unsafe_allow_html=True)
            result_placeholder = st.empty()

        while True:
            try:
                self.process_frame()
                self.process_hands()
                self.identify_fingers()
                self.handle_drawing_mode()
                self.blend_canvas_with_feed()
                stframe.image(self.img, channels="RGB")

                if sum(self.fingers) == 2 and self.fingers[1] == self.fingers[2] == 1:
                    result = self.analyze_image_with_genai()
                    result_placeholder.write(f"Result: {result}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
                break

        self.cap.release()
        cv2.destroyAllWindows()


# Run the app
try:
    calc = Calculator()
    calc.streamlit_config()
    calc.main()
except Exception as e:
    st.error(f"Application failed with error: {e}")
