# Dockerfile - ambiente do projeto de Redes Neurais 

# Todo mundo com o mesmo ambiente :)

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# build-essential: rede de segurança, caso o pip precise compilar alguma
# dependencia em uma plataforma sem wheel pronta (raro, mas evita builds
# quebrados em maquinas menos comuns)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--ServerApp.token=''"]
