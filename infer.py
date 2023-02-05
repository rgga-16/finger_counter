import cv2
import torch
import numpy as np
from models import CustomClassifier
from dataset import data_transforms
import config
from PIL import Image

device = config.device

# Load your trained PyTorch image classifier
model = CustomClassifier().to(device)
model.load_state_dict(torch.load('./weights/weights.pt'))
model.eval()

handtypes = [0.0,1.0] # Left=0.0, Right=1.0
counts = [0.0,1.0,2.0,3.0,4.0,5.0]

# Start the webcam stream
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to a tensor and normalize the pixel values
    frame_pil = Image.fromarray(frame)
    frame_tensor = data_transforms['test'](frame_pil).to(device)
    # frame_tensor = torch.from_numpy(frame).float()
    # frame_tensor = frame_tensor.permute(2, 0, 1)
    # frame_tensor = frame_tensor / 255

    # Pass the frame tensor through the model for inference
    pred_types, pred_counts = model(frame_tensor.unsqueeze(0))

    # Convert the output to a numpy array and find the class with the highest score
    output_np = pred_counts.detach().cpu().numpy()
    class_index = np.argmax(output_np)

    predicted_class = counts[class_index]

    # Display the class index on the frame
    cv2.putText(frame, str(predicted_class), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('frame', frame)

    # Check if the user pressed the 'q' key to stop the stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
