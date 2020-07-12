# Project Write-Up

You can use this document as a template for providing your project write-up. However, if you
have a different format you prefer, feel free to use it as long as you answer all required
questions.

NOTICE: most of the code is inspired by the udacity classes code.

## Explaining Custom Layers

Custom layers are the layers that are not included in the layers list, so if a model includes a layer that isn't on the layers list. it will be considered a custo layers

While doing the inference on those two models, there was no existence of custom layers. However if there were any, they should be replaced by a plugged in new layer known to the engine.
## Comparing Model Performance

I used accurcy, speed and least infeence time to comapre between models, for which I used OpenVINO DL workbench to analyze the inference behavior on each model fo a generated dataset.

The difference between model accuracy pre- and post-conversion was...

** since the vieos and imgaes are unnotated datasets, then we cannot calculate accuracy. In this case I depended on the observation fo FP (False positive) and FN (False negative). 

Also the accurcy was affected much by the confidence threshold parameter tunning and the leave time threshold, as the following table shows:

MODEL Name | conf threshold | time threshold | FP | FN | accuracy | count
-----------|----------------|----------------|----|----|----------|------
ssd_mobilenet_v1_ppn_2018 | 0.6 | 3 | No| No| 100 | 16
ssd_mobilenet_v2_coco_2018 | 0.6 |3|No| 1 | 5/6| 17
ssd_mobilenet_v1_ppn_2018 | 0.2 | 3 | 100| No| 0 | 100
ssd_mobilenet_v2_coco_2018 | 0.2 |3|100| - | 0| 100
ssd_mobilenet_v1_ppn_2018 | 0.7 | 5 | No| No| 100 | 12
ssd_mobilenet_v2_coco_2018 | 0.7 |5|No| No | 6/6| 13
ssd_mobilenet_v1_ppn_2018 | 0.7 | 9 | No| No| 100 | 6
ssd_mobilenet_v2_coco_2018 | 0.8 |9|No| 1| 6/6| 7

So, ssd_mobilenet_v1_ppn_2018 was sufficient to fullfill the requirements since it was accurate, simple to convert and gave the right count after tunning the parameters a little. 

##  the comparisons before and after conversion:

MODEL Name | pre size | post size  | Pre_inffer time | post_infer time
-----------|----------|----------|----|-----------
ssd_mobilenet_v1_ppn_2018 | 10,540KB | 75 KB | 37 ms | 8-9 ms 
ssd_mobilenet_v2_coco_2018 | 66.5 MB | 109 KB | not measured | 12-13 ms 

## Assess Model Use Cases

It can be used to ogaize the social distancing nd the room  maximum allowance capacity fo COVID-19.

Also, it can be used in business offices in normal conditions to help tack the number of customers per day, which can be used fo better statisical measures.

Also, it can be used for service shelters to measure how many people an employee can serve per day.

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows...

Lighting> the lighting of the video made the fifth person in he video, somehow difficult to detect, since the color of the clothes and the black hair were similar to the room colors.  The camera was fixed, so it was easz to detect moving objects, however if it was motile, it would have been difficult to detect people with these models. Also, changing the focal length of the camera will definitely equire you to change the confidence threshold. 

## Model Research

[This heading is only required if a suitable model was not found after trying out at least three
different models. However, you may also use this heading to detail how you converted 
a successful model.]

In investigating potential people counter models, I tried each of the following three models:

- Model 1: [ssd_mobilenet_v1_ppn_coco ]
  - [http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03.tar.gz]
  - I converted the model to an Intermediate Representation with the following arguments...
  <pre><code>
  cd ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03
  
  python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json

</code></pre>
  - The model was insufficient for the app because, it has some false positives at the beginning since it was giving a higher count of people always.
  - I tried to improve the model for the app by tuning the confidence index and tuning the threshold leave time for inference
  
- Model 2: [Name]
  - [http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz]

  <pre><code>
  
  python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config ssd_mobilenet_v2_coco_2018_03_29/pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json

</code></pre>

- Model 3: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...
