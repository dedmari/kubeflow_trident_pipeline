# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import kfp.dsl as dsl
import datetime
import os
from kubernetes import client as k8s_client


# Modify image='<image>' in each op to match IMAGE in the build.sh of its corresponding component
# Modified by Muneer
# Defining Images as variable.
preprocess_image='muneer7589/k_pipeline_preprocess:latest'
train_image='muneer7589/k_pipeline_train:latest'
inference_server_launcher_image='muneer7589/k_pipeline_inference:latest'
webapp_launcher_image='muneer7589/k_pipeline_webapp_launcher:latest'
 
# TODO: Create configuration file for storing images and make that as general template for generating different pipelines


def PreprocessOp(name, input_dir, output_dir):
    return dsl.ContainerOp(
        name=name,
        image=preprocess_image,
        arguments=[
            '--input_dir', input_dir,
            '--output_dir', output_dir,
        ],
        file_outputs={'output': '/output.txt'}
    )


def TrainOp(name, input_dir, output_dir, model_name, model_version, epochs):
    return dsl.ContainerOp(
        name=name,
        image=train_image,
        arguments=[
            '--input_dir', input_dir,
            '--output_dir', output_dir,
            '--model_name', model_name,
            '--model_version', model_version,
            '--epochs', epochs
        ],
        file_outputs={'output': '/output.txt'}
    )


def InferenceServerLauncherOp(name, input_dir, trtserver_name):
    print("inside inference server ContainerOp and value of name is: ", name)
    return dsl.ContainerOp(
        name=name,
        image=inference_server_launcher_image,
        arguments=[
            '--trtserver_name', trtserver_name,
            '--model_path', input_dir,
        ],
        file_outputs={'output': '/output.txt'}
    )


def WebappLauncherOp(name, trtserver_name, model_name, model_version, webapp_prefix, webapp_port):
    print("Inside WeappLauncherOp and the value of name is: ", name)
    return dsl.ContainerOp(
        name=name,
        image=webapp_launcher_image,
        arguments=[
            '--workflow_name', '{{workflow.name}}',
            '--trtserver_name', trtserver_name,
            '--model_name', model_name,
            '--model_version', str(model_version),
            '--webapp_prefix', webapp_prefix,
            '--webapp_port', str(webapp_port)
        ],
        file_outputs={}
    )


@dsl.pipeline(
    name='resnet_cifar10_pipeline',
    description='Demonstrate an end-to-end training & serving pipeline using ResNet and CIFAR-10'
)
def resnet_pipeline(
    raw_data_dir='/mnt/workspace/raw_data',
    processed_data_dir='/mnt/workspace/processed_data',
    model_dir='/mnt/workspace/saved_model',
    epochs=50,
    trtserver_name='trtis',
    model_name='resnet_graphdef',
    model_version=1,
    webapp_prefix='webapp',
    webapp_port=80
):

    persistent_volume_name = 'pipeline-workspace'
    persistent_volume_path = '/mnt/workspace'

    op_dict = {}

    op_dict['preprocess'] = PreprocessOp(
        'preprocess', raw_data_dir, processed_data_dir)

    op_dict['train'] = TrainOp(
        'train', op_dict['preprocess'].output, model_dir, model_name, model_version, epochs)

    op_dict['deploy_inference_server'] = InferenceServerLauncherOp(
        'deploy_inference_server', op_dict['train'].output, trtserver_name)

    op_dict['deploy_webapp'] = WebappLauncherOp(
        'deploy_webapp', op_dict['deploy_inference_server'].output, model_name, model_version, webapp_prefix, webapp_port)

    for _, container_op in op_dict.items():
        container_op.add_volume(k8s_client.V1Volume(
            persistent_volume_claim=k8s_client.V1PersistentVolumeClaimVolumeSource(
                claim_name=persistent_volume_name),
            name=persistent_volume_name))
        container_op.add_volume_mount(k8s_client.V1VolumeMount(
            mount_path=persistent_volume_path,
            name=persistent_volume_name))


if __name__ == '__main__':
    import kfp.compiler as compiler
    compiler.Compiler().compile(resnet_pipeline, __file__ + '.tar.gz')
