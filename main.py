import cv2
import mediapipe as mp
import gxipy as gx

mp_drawing = mp.solutions.drawing_utils
mp_objectron = mp.solutions.objectron

# # For static images:
# IMAGE_FILES = []
# with mp_objectron.Objectron(static_image_mode=True,
#                             max_num_objects=5,
#                             min_detection_confidence=0.5,
#                             model_name='Shoe') as objectron:
#   for idx, file in enumerate(IMAGE_FILES):
#     image = cv2.imread(file)
#     # Convert the BGR image to RGB and process it with MediaPipe Objectron.
#     results = objectron.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

#     # Draw box landmarks.
#     if not results.detected_objects:
#       print(f'No box landmarks detected on {file}')
#       continue
#     print(f'Box landmarks of {file}:')
#     annotated_image = image.copy()
#     for detected_object in results.detected_objects:
#       mp_drawing.draw_landmarks(
#           annotated_image, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
#       mp_drawing.draw_axis(annotated_image, detected_object.rotation,
#                            detected_object.translation)
#       cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)

# For webcam input:

device_manager = gx.DeviceManager()
device = device_manager.update_device_list()
device = device_manager.open_device_by_index(1)



with mp_objectron.Objectron(static_image_mode=False,
                            max_num_objects=5,
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.99,
                            model_name='Shoe') as objectron:
  
  while True:

    try:
      if device is None:
        print("No U3V device found")
        exit() 
      
      device.stream_on()
      #get raw data
      raw_image = device.data_stream[0].get_image()
      #convert to numpy array
      raw_image = raw_image.convert("RGB")
      image_data = raw_image.get_numpy_array()
      # To improve performance, optionally mark the image as not writeable to
      # pass by reference.
      image_data.flags.writeable = False
      image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
      results = objectron.process(image_data)
      # Draw the box landmarks on the image.
      image_data.flags.writeable = True
      image_data = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)
      results = objectron.process(image_data)

      # Draw the box landmarks on the image.
      image_data.flags.writeable = True
      image_data = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)
      if results.detected_objects:
        for detected_object in results.detected_objects:
          mp_drawing.draw_landmarks(
            image_data, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS
          )
          mp_drawing.draw_axis(
            image_data, detected_object.rotation,detected_object.translation
          )
      # Flip the image horizontally for a selfie-view display.
      cv2.imshow('MediaPipe Objectron', cv2.flip(image_data, 1))
      if cv2.waitKey(1) & 0XFF == ord('q'):
        break
    except Exception as e:
      print(f"error occured {e}")
  device.stream_off()
  cv2.destroyAllWindows()

  
          

  
#   try:
#     if device is None:
#       print("No U3V device found")
#       exit()
#       device_manager.update_all_device_list()
#     device = device_manager.open_device_by_index(1)
#     device.stream_on()
    
#     #get raw data
#     raw_image = device.data_stream[0].get_image()
#     #convert to numpy array
#     raw_image = raw_image.convert("RGB")
#     image_data = raw_image.get_numpy_array()

#     # To improve performance, optionally mark the image as not writeable to
#     # pass by reference.
#     image_data.flags.writeable = False
#     image_data = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     results = objectron.process(image_data)

#     # Draw the box landmarks on the image.
#     image_data.flags.writeable = True
#     image_data = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)
#     if results.detected_objects:
#         for detected_object in results.detected_objects:
#             mp_drawing.draw_landmarks(
#               image_data, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
#             mp_drawing.draw_axis(image_data, detected_object.rotation,
#                                  detected_object.translation)
#     # Flip the image horizontally for a selfie-view display.
#     cv2.imshow('MediaPipe Objectron', cv2.flip(image_data, 1))
#     if cv2.waitKey(1) & 0XFF == ord('q'):
#       break
#   except Exception as e:
#     print(f"error occured {e}")
#   device.stream_off()
#   cv2.destroyAllWindows()
