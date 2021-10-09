FROM python:3.8-slim-buster
COPY . /srv
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
EXPOSE 5000
WORKDIR /srv
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc
RUN pip install -r requirements.txt
RUN chmod +x /usr/local/bin/docker-entrypoint.sh
COPY src/train.py /srv/train.py
COPY src/service.py /srv/service.py
COPY data/items.csv /srv/items.csv
COPY data/users.csv /srv/users.csv
COPY data/user_item.csv /srv/user_item.csv
ENTRYPOINT ["docker-entrypoint.sh"]