"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2
import logging as log
import paho.mqtt.client as mqtt
from sys import platform 
from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

'''if platform == "linux" or platform == "linux2":
    CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"
    CODEC = 0x00000021
elif platform == "darwin":
    CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension.dylib"
    CODEC = cv2.VideoWriter_fourcc('M','J','P','G')
else:
    print("Unsupported OS.")
    exit(1)
'''

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    parser.add_argument("-tt", "--time_threshold", type=float, default=4,
                        help="Leave time threshold"
                        "(5 secs by default)")
    return parser
 

def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client

   ## Draw bounding boxes onto the frame.
def handle_stream (args):
    if args.input == 'CAM':
        valid_input = 0

    # Checks for input image
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp') :
        single_image_mode = True
        valid_input = args.input

    # Checks for video file
    else:
        valid_input = args.input
        assert os.path.isfile(args.input), "file doesn't exist"
    return valid_input

def preprocess (frame, input_shape):
    image = cv2.resize(frame, (input_shape[3], input_shape[2]))
    image_p = image.transpose((2, 0, 1))
    image_p = image_p.reshape(1, *image_p.shape)
    return image_p

def assess_scene (args, mqttclient, duration_report , leave_report, current_duration, total_count, current_counter, previous_counter, previous_duration,current_count):
    container = 0
    ''' the pogram starts with curent count = the current counte = 0, and then goes to the else part of the if condition. if the current counter equals the current (instant) count, increase the curent duration by 1 and check if it is equal to the threshold yet. if yes, then the set the leave_report to the curent counter, then check if the curent counter less than the previous counter, if yes. then the duration leave report will be equal the previous (terminated and stored) duration, else if previous_count is less than the current_counter, then total count will be current_counter - previous_counter. on the other hand if the current_duration is geater than the threshold , and if the the duration exceeds 6 secs then publish the notice that the person exceeds the 6 secs duration  '''
    tt = args .time_threshold 
    if current_count == current_counter:
        current_duration = current_duration + 1
        if current_duration == tt:
            leave_report = current_counter
            if current_counter > previous_counter:
                container = current_counter - previous_counter
                total_count +=  container 
            elif current_counter < previous_counter:
                duration_report = previous_duration
        elif current_duration >  tt:
            leave_report = current_counter            
    else:
        '''  if current count doesn't equal  the counter, then make it a previous counter and put the current count in the current counter. which means the number of people has change so change the current counter according to the new count. then check i f the current duration is greater than the leave time threshold, if yes start counting for another duration and move the current_counter to previous_counter, otherwise add the previous duaration to the current duration and set the previous uration to zero '''
        if current_duration < tt:
            previous_counter = current_counter
            current_counter = current_count 
            current_duration = previous_duration + current_duration
            previous_duration = 0 
        else:
            previous_counter = current_counter
            current_counter = current_count
            previous_duration = current_duration
            current_duration = 0
            
    return duration_report , leave_report, current_duration, total_count, current_counter, previous_counter, previous_duration,current_count

def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    plugin = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold
    model = args.model
    cpu_extension = args.cpu_extension
    device = args.device
    tt = args.time_threshold
        
    ### TODO: Load the model through `infer_network` ###
    plugin.load_model(model, device, cpu_extension)
    network_shape = plugin.get_input_shape()
    

    ### TODO: Handle the input stream ###
    valid_input = handle_stream (args)

    ### TODO: Loop until stream is over ###
    cap = cv2.VideoCapture(valid_input)
    cap.open(valid_input)

    w = int(cap.get(3))
    h = int(cap.get(4))

    input_shape = network_shape
    leave_report = 0
    current_counter = 0    
    current_duration = 0
    previous_duration = 0
    total_count = 0
    request_id = 0
    previous_counter = 0
    while cap.isOpened():
        ### TODO: Read from the video capture ###
        t0 = time.time()
        flag, frame = cap.read()
        t1 = time.time()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        ### TODO: Pre-process the image as needed ###
        p_frame= preprocess (frame, input_shape)

        ### TODO: Start asynchronous inference for specified request ###
        plugin.exec_net(p_frame)
        

        ### TODO: Wait for the result ###
        if plugin.wait() == 0:

            ### TODO: Get the results of the inference request ###
            result = plugin.get_output()
            current_count  = 0
            duration_report =0
            ### TODO: Extract any desired stats from the results ###
            for box in result[0][0]: 
                conf = box[2]
                if conf >= prob_threshold:
                    current_count += 1
                    xmin = int(box[3] * w)
                    ymin = int(box[4] * h)
                    xmax = int(box[5] * w)
                    ymax = int(box[6] * h)
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
                    
            
            ###  TODO: Calculate and send relevant information on ###
            duration_report , leave_report, current_duration, total_count, current_counter, previous_counter, previous_duration,current_count = assess_scene (args,client,duration_report , leave_report, current_duration, total_count, current_counter, previous_counter, previous_duration,current_count)
            ### current_count, total_count and duration to the MQTT server ###
             ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            client.publish('person ',
                           payload=json.dumps({
                               'count': leave_report, 'total': total_count}))
            
            if duration_report > 0:
                client.publish('person/duration',
                               payload=json.dumps({'duration': duration_report}))

        ### TODO: Send the frame to the FFMPEG server ###

        ### TODO: Write an output image if `single_image_mode` ###
        frame = cv2.resize(frame, (768, 432))
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()
        
        ### TODO: Write an output image if `single_image_mode` ###
        '''if single_image_mode:
            cv2.imwrite("output_img.jpg", frame)'''

    cap.release()
    cv2.destroyAllWindows()
        


def main():
    """
    Load the network and  parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server 
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
 