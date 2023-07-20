FROM python:3.9

ARG USER=user
ARG ID=1000
ARG REQUIREMENTS_TXT="requirements.txt"
ARG HOME_DIR="/home/$USER"

RUN groupadd -g $ID $USER && useradd -g $ID -m -u $ID -s /bin/bash $USER
WORKDIR $HOME_DIR
USER $USER

COPY --chown=$ID:$ID $REQUIREMENTS_TXT .
RUN pip3 install -r $REQUIREMENTS_TXT

COPY --chown=$ID:$ID main.py main.py
COPY --chown=$ID:$ID src/ src/
COPY --chown=$ID:$ID chroma_db/ chroma_db/
COPY --chown=$ID:$ID config/ config/

EXPOSE 8501

ENTRYPOINT ["python", "-m", "streamlit", "run", "main.py"]