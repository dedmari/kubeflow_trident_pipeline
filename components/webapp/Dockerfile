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

# Editted by Muneer
# FROM base-trtis-client
FROM muneer7589/tensorrtserver_client

# pip3 installation included by Muneer
RUN apt-get install -y python3-pip

# extacting packages for tensorrt server
RUN tar -xvf /workspace/v1.2.0.clients.tar.gz

# installating extracted packages, numpy and pillow by Muneer
RUN pip3 install --user --upgrade python/tensorrtserver-*.whl numpy pillow 

# numpy installation included by Muneer
# RUN pip3 install numpy

# grpc installation included by Muneer
RUN pip3 install grpcio

RUN pip3 install flask
ADD src /workspace/web_server
WORKDIR /workspace/web_server
EXPOSE 8080

ENTRYPOINT ["python3", "flask_server.py"]
