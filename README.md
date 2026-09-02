# Ambiente do projeto — Redes Neurais 

**Bibliotecas Python**

| `numpy` | Todas as operações de matriz: forward, backprop, gradientes |
| `scipy` | `optimize.minimize` (método `'CG'`) para o gradiente conjugado — é uma rotina de otimização genérica, não um framework de rede neural |
| `matplotlib` | Dígitos mal classificados, unidades escondidas, curva de aprendizado, erro × λ |
| `pandas` | Ler `imageMNIST.csv` / `labelMNIST.csv` de forma prática (opcional — `numpy.genfromtxt` também resolve) |
| `jupyterlab` | Interface para o grupo todo rodar/editar os notebooks |

O split treino/validação/teste também não precisa de `scikit-learn`: dá
para embaralhar os índices com `numpy.random` e fatiar o array.

## Arquivos

| Arquivo | Para que serve |
|---|---|
| `Dockerfile` | Define a imagem: Python 3.11 + bibliotecas acima + JupyterLab |
| `requirements.txt` | Bibliotecas que todo mundo vai usar |
| `docker-compose.yml` | Sobe o container, mapeia a porta do Jupyter, define CPU/RAM |
| `.env.example` | Modelo para cada um configurar os próprios limites de recursos |
| `.gitignore` | Evita subir cache, `.env` pessoal etc. para o repositório do grupo |

## Como usar

1. Instale o Docker:
   - **Windows/macOS**: Docker Desktop.
   - **Linux**: Docker Engine + plugin `docker-compose-plugin`.
2. Coloque `imageMNIST.csv` e `labelMNIST.csv` na pasta do projeto (mesmo
   nível do `Dockerfile`), ou numa subpasta tipo `data/` — como preferirem.
3. *(Opcional)* copie `.env.example` para `.env` e ajuste `CPU_LIMIT` /
   `MEM_LIMIT` para o seu notebook (ver seção de recursos abaixo).
4. Na pasta do projeto:
   ```bash
   docker compose up --build
   ```
5. Abra `http://localhost:8888` — o JupyterLab abre direto, sem pedir token.
6. Crie/edite os notebooks normalmente. Tudo que está na pasta do projeto
   (no seu computador) aparece dentro do container em `/workspace`, e
   vice-versa: os notebooks continuam lá depois de fechar o container, e
   podem ser commitados no git normalmente.

Para desligar: `Ctrl+C` no terminal e depois `docker compose down`.

Preferir rodar `.py` em vez de notebook? Dá pra abrir um terminal dentro do
container já em execução:
```bash
docker compose exec nn-mnist bash
```

## Onde ficam os recursos do container (CPU / RAM)

A seleção de recursos fica no `docker-compose.yml`, nestas três linhas:

```yaml
cpus: "${CPU_LIMIT:-2}"
mem_limit: "${MEM_LIMIT:-2g}"
mem_reservation: "${MEM_RESERVATION:-1g}"
```

(De propósito usamos `cpus` / `mem_limit` / `mem_reservation` direto no
serviço, e não dentro de um bloco `deploy:` — esses últimos só valem, por
padrão, quando se usa `docker stack deploy` com Swarm; os atributos acima
funcionam com o `docker compose up` normal, que é o que o grupo vai usar.)

Os padrões (2 CPUs / 2 GB de limite) já são confortáveis: a base usada é
pequena (alguns milhares de exemplos, 400 atributos), então até um notebook
mais fraco roda sem problema. Cada pessoa do grupo pode subir ou baixar
esses valores para a própria máquina criando um `.env` local (a partir do
`.env.example`) — como o `.env` não é commitado, cada um mantém sua própria
configuração sem gerar conflito no repositório compartilhado.

## E a placa de vídeo (GPU)?

Não é necessária para este projeto. O código usa numpy/scipy "puro" (é a
própria exigência do professor), e essas contas rodam na CPU. Como a base é
pequena, treinar a rede — mesmo repetindo para vários λ na Parte II — leva
segundos a poucos minutos; GPU não traria ganho nenhum aqui, a menos que a
implementação trocasse `numpy` por `cupy`, o que o enunciado não pede.

Ainda assim, se alguém do grupo tiver GPU NVIDIA e quiser mexer nisso por
conta própria, há uma linha comentada em `docker-compose.yml`:
```yaml
# gpus: all
```
Basta descomentar (requer Docker Compose 2.30+ e os drivers NVIDIA no
host — no Linux, com o NVIDIA Container Toolkit; no Windows, via WSL2).

## Observações

- **Windows**: para melhor performance, mantenha a pasta do projeto dentro
  do sistema de arquivos do WSL2 (não direto em `C:\Users\...`), com o
  Docker Desktop configurado no backend WSL2.
- Todos os comandos usam `docker compose` (sem hífen — a versão atual,
  incluída no Docker Desktop). Se ainda houver o `docker-compose` antigo
  (com hífen) instalado, prefira o `docker compose` novo.
