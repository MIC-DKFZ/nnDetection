FROM nndetection:0.1
# inference file to run predictions

ARG task
ARG model
ARG fold

ENV ENVTASK=$task ENVMODEL=$model ENVFOLD=$fold det_results=/opt/results
ENV det_data=/opt/data
ENV det_models=/opt/models

RUN echo ${ENVTASK} ${ENVMODEL} ${ENVFOLD} \
    && mkdir -p ${det_models}/${ENVTASK}/${ENVMODEL}/${ENVFOLD} \
    && mkdir -p ${det_data}/${ENVTASK}/raw_splitted/imagesTs \
    && mkdir -p ${det_results}

COPY ./${ENVFOLD}/* ${det_models}/${ENVTASK}/${ENVMODEL}/${ENVFOLD}/
COPY scripts/inference.sh .
COPY scripts/convert.py .
