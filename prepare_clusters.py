"""
Gera embeddings visuais OpenCLIP, projeção UMAP e clusters HDBSCAN
a partir dos thumbnails preparados para o dashboard.

Entrada:
    dashboard_data.json

Saídas:
    clusters.json
        Arquivo leve utilizado pelo dashboard e pelo GitHub Pages.

    cluster_embeddings_openclip.npy
        Embeddings visuais completos.
        Este arquivo serve para reutilização local e não precisa
        ser publicado no GitHub Pages.

    cluster_embedding_items_openclip.json
        Manifesto que registra:
            - modelo OpenCLIP;
            - checkpoint utilizado;
            - ordem dos itens nos embeddings.

Pipeline:
    Thumbnails WebP
        -> OpenCLIP image embeddings
        -> normalização L2
        -> PCA opcional
        -> UMAP 2D
        -> HDBSCAN

Instalação:
    pip uninstall -y open_clip

    pip install \
        open_clip_torch \
        torch \
        torchvision \
        pillow \
        numpy \
        scikit-learn \
        umap-learn \
        hdbscan \
        tqdm

Primeira execução:
    python prepare_clusters.py \
        --dashboard-data dashboard_data.json \
        --output clusters.json \
        --embeddings-output cluster_embeddings_openclip.npy \
        --items-output cluster_embedding_items_openclip.json \
        --model-name ViT-B-32 \
        --pretrained laion2b_s34b_b79k \
        --batch-size 8 \
        --min-cluster-size 12 \
        --min-samples 5

Reutilizar os embeddings e testar outros parâmetros:
    python prepare_clusters.py \
        --dashboard-data dashboard_data.json \
        --output clusters.json \
        --embeddings-output cluster_embeddings_openclip.npy \
        --items-output cluster_embedding_items_openclip.json \
        --model-name ViT-B-32 \
        --pretrained laion2b_s34b_b79k \
        --reuse-embeddings \
        --min-cluster-size 8 \
        --min-samples 3

Observações:
    - O clustering usa os itens presentes na chave "gallery" de
      dashboard_data.json.
    - Se prepare_dashboard.py gerou 1.500 itens, o clustering usará
      esses mesmos itens, exceto imagens ausentes ou inválidas.
    - Não reutilize os antigos embeddings DINOv2.
"""

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import hdbscan
import numpy as np
import open_clip
import torch
import umap
from PIL import Image, ImageOps
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from tqdm import tqdm


# ============================================================
# CONFIGURAÇÃO
# ============================================================

OPENCLIP_MODEL_NAME = "ViT-B-32"
OPENCLIP_PRETRAINED = "laion2b_s34b_b79k"

DEFAULT_BATCH_SIZE = 8
DEFAULT_MIN_CLUSTER_SIZE = 12
DEFAULT_MIN_SAMPLES = 5
DEFAULT_RANDOM_STATE = 42

# O OpenCLIP ViT-B-32 gera embeddings de 512 dimensões.
# PCA reduz o custo do HDBSCAN e pode diminuir ruído.
DEFAULT_PCA_COMPONENTS = 25

# Configuração padrão do UMAP.
DEFAULT_UMAP_NEIGHBORS = 15
DEFAULT_UMAP_MIN_DIST = 0.08


# ============================================================
# ARGUMENTOS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Gera embeddings OpenCLIP, projeção UMAP e "
            "clusters HDBSCAN para as imagens do dashboard."
        )
    )

    parser.add_argument(
        "--dashboard-data",
        type=Path,
        default=Path("dashboard_data.json"),
        help=(
            "JSON produzido pelo script prepare_dashboard.py."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("clusters.json"),
        help=(
            "Arquivo JSON final utilizado pelo dashboard."
        ),
    )

    parser.add_argument(
        "--embeddings-output",
        type=Path,
        default=Path(
            "cluster_embeddings_openclip.npy"
        ),
        help=(
            "Arquivo NumPy usado para armazenar os embeddings."
        ),
    )

    parser.add_argument(
        "--items-output",
        type=Path,
        default=Path(
            "cluster_embedding_items_openclip.json"
        ),
        help=(
            "Manifesto com a configuração do OpenCLIP "
            "e a ordem dos itens nos embeddings."
        ),
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default=OPENCLIP_MODEL_NAME,
        help=(
            "Arquitetura OpenCLIP usada para gerar embeddings."
        ),
    )

    parser.add_argument(
        "--pretrained",
        type=str,
        default=OPENCLIP_PRETRAINED,
        help=(
            "Checkpoint pré-treinado do OpenCLIP."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=(
            "Quantidade de imagens processadas por lote. "
            "Em Apple Silicon, comece com 8."
        ),
    )

    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=DEFAULT_MIN_CLUSTER_SIZE,
        help=(
            "Tamanho mínimo de um cluster no HDBSCAN."
        ),
    )

    parser.add_argument(
        "--min-samples",
        type=int,
        default=DEFAULT_MIN_SAMPLES,
        help=(
            "Parâmetro min_samples do HDBSCAN. "
            "Valores maiores produzem clustering mais conservador."
        ),
    )

    parser.add_argument(
        "--max-items",
        type=int,
        default=0,
        help=(
            "Quantidade máxima de itens processados. "
            "Use 0 para processar toda a galeria."
        ),
    )

    parser.add_argument(
        "--pca-components",
        type=int,
        default=DEFAULT_PCA_COMPONENTS,
        help=(
            "Número de componentes PCA antes do clustering. "
            "Use 0 para desativar o PCA."
        ),
    )

    parser.add_argument(
        "--umap-neighbors",
        type=int,
        default=DEFAULT_UMAP_NEIGHBORS,
        help=(
            "Número de vizinhos usado pelo UMAP."
        ),
    )

    parser.add_argument(
        "--umap-min-dist",
        type=float,
        default=DEFAULT_UMAP_MIN_DIST,
        help=(
            "Distância mínima usada pelo UMAP, entre 0 e 1."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help=(
            "Semente usada para reprodutibilidade."
        ),
    )

    parser.add_argument(
        "--reuse-embeddings",
        action="store_true",
        help=(
            "Reutiliza os embeddings e a ordem de itens "
            "salvos em uma execução anterior."
        ),
    )

    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error(
            "--batch-size deve ser pelo menos 1"
        )

    if args.min_cluster_size < 2:
        parser.error(
            "--min-cluster-size deve ser pelo menos 2"
        )

    if args.min_samples < 1:
        parser.error(
            "--min-samples deve ser pelo menos 1"
        )

    if args.max_items < 0:
        parser.error(
            "--max-items deve ser maior ou igual a zero"
        )

    if args.pca_components < 0:
        parser.error(
            "--pca-components deve ser maior ou igual a zero"
        )

    if args.umap_neighbors < 2:
        parser.error(
            "--umap-neighbors deve ser pelo menos 2"
        )

    if not 0 <= args.umap_min_dist <= 1:
        parser.error(
            "--umap-min-dist deve estar entre 0 e 1"
        )

    return args


# ============================================================
# UTILITÁRIOS
# ============================================================

def configurar_sementes(seed):
    """
    Configura sementes para aumentar a reprodutibilidade.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def detectar_device():
    """
    Escolhe automaticamente CUDA, MPS ou CPU.
    """
    if torch.cuda.is_available():
        print("Usando GPU NVIDIA via CUDA")
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        print("Usando Apple Silicon GPU via MPS")
        return torch.device("mps")

    print("Usando CPU")
    return torch.device("cpu")


def carregar_json(caminho):
    caminho = Path(caminho)

    if not caminho.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {caminho}"
        )

    with caminho.open(
        "r",
        encoding="utf-8",
    ) as arquivo:
        return json.load(arquivo)


def salvar_json(
    caminho,
    dados,
):
    caminho = Path(caminho)

    caminho.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    caminho.write_text(
        json.dumps(
            dados,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def numero_finito(
    valor,
    padrao=0.0,
):
    """
    Converte um valor para float e impede NaN ou infinito.
    """
    try:
        resultado = float(valor)

        if math.isfinite(resultado):
            return resultado

    except (TypeError, ValueError):
        pass

    return padrao


def normalizar_embeddings(
    embeddings,
):
    """
    Aplica normalização L2 em cada vetor.
    """
    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    normas = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True,
    )

    normas[normas == 0] = 1

    return (
        embeddings / normas
    ).astype(np.float32)


def limpar_memoria_device(
    device,
):
    """
    Tenta liberar caches de CUDA ou MPS entre lotes.
    """
    if device.type == "cuda":
        torch.cuda.empty_cache()

    elif device.type == "mps":
        try:
            torch.mps.empty_cache()
        except AttributeError:
            pass


def carregar_imagem(caminho):
    """
    Abre uma imagem, corrige sua orientação e converte para RGB.
    """
    with Image.open(caminho) as imagem:
        imagem = ImageOps.exif_transpose(
            imagem
        )

        if imagem.mode in (
            "RGBA",
            "LA",
        ):
            rgba = imagem.convert("RGBA")

            fundo = Image.new(
                "RGBA",
                rgba.size,
                (255, 255, 255, 255),
            )

            fundo.alpha_composite(rgba)

            return fundo.convert("RGB")

        return imagem.convert("RGB")


# ============================================================
# PREPARAÇÃO DOS ITENS
# ============================================================

def preparar_registros(
    dashboard,
    dashboard_directory,
    max_items,
):
    """
    Seleciona os itens da galeria que possuem thumbnails válidos.

    Caminhos relativos presentes no dashboard_data.json são
    resolvidos em relação ao diretório desse arquivo.
    """
    registros = dashboard.get(
        "gallery",
        [],
    )

    if not isinstance(
        registros,
        list,
    ):
        raise ValueError(
            "A chave 'gallery' de dashboard_data.json "
            "precisa conter uma lista."
        )

    registros_validos = []
    imagens_ausentes = 0

    for registro in registros:
        if not isinstance(
            registro,
            dict,
        ):
            continue

        crop_image = registro.get(
            "crop_image"
        )

        if not crop_image:
            imagens_ausentes += 1
            continue

        image_path = Path(
            crop_image
        ).expanduser()

        if not image_path.is_absolute():
            image_path = (
                dashboard_directory
                / image_path
            )

        image_path = image_path.resolve()

        if not image_path.exists():
            print(
                "Warning: thumbnail não encontrado: "
                f"{image_path}"
            )

            imagens_ausentes += 1
            continue

        if not image_path.is_file():
            print(
                "Warning: o caminho não é um arquivo: "
                f"{image_path}"
            )

            imagens_ausentes += 1
            continue

        registro_copia = dict(
            registro
        )

        registro_copia[
            "_image_path"
        ] = str(image_path)

        registros_validos.append(
            registro_copia
        )

        if (
            max_items > 0
            and len(registros_validos)
            >= max_items
        ):
            break

    print(
        "Itens válidos para clustering: "
        f"{len(registros_validos)}"
    )

    if imagens_ausentes:
        print(
            "Itens ignorados por imagem ausente: "
            f"{imagens_ausentes}"
        )

    return registros_validos


# ============================================================
# EMBEDDINGS OPENCLIP
# ============================================================

def extrair_embeddings(
    registros,
    preprocess,
    model,
    device,
    batch_size,
):
    """
    Gera embeddings visuais com o image encoder do OpenCLIP.

    Para cada imagem:
        1. abre e converte para RGB;
        2. aplica o preprocess oficial do modelo;
        3. executa model.encode_image();
        4. normaliza o vetor com L2.
    """
    embeddings = []
    registros_processados = []

    for inicio in tqdm(
        range(
            0,
            len(registros),
            batch_size,
        ),
        desc="Extraindo embeddings OpenCLIP",
    ):
        lote = registros[
            inicio:inicio + batch_size
        ]

        tensores = []
        lote_valido = []

        for registro in lote:
            image_path = Path(
                registro["_image_path"]
            )

            try:
                imagem = carregar_imagem(
                    image_path
                )

                tensor = preprocess(
                    imagem
                )

                tensores.append(
                    tensor
                )

                lote_valido.append(
                    registro
                )

            except Exception as erro:
                print(
                    "Warning: não foi possível processar "
                    f"{image_path}: {erro}"
                )

        if not tensores:
            continue

        image_batch = torch.stack(
            tensores,
            dim=0,
        ).to(device)

        with torch.inference_mode():
            batch_embeddings_tensor = (
                model.encode_image(
                    image_batch
                )
            )

            # Garante float32, inclusive quando o modelo
            # estiver usando precisão reduzida.
            batch_embeddings_tensor = (
                batch_embeddings_tensor.float()
            )

            # Normalização L2.
            batch_embeddings_tensor = (
                batch_embeddings_tensor
                / batch_embeddings_tensor.norm(
                    dim=-1,
                    keepdim=True,
                ).clamp(
                    min=1e-12
                )
            )

            batch_embeddings = (
                batch_embeddings_tensor
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

        embeddings.append(
            batch_embeddings
        )

        registros_processados.extend(
            lote_valido
        )

        del image_batch
        del batch_embeddings_tensor
        del batch_embeddings
        del tensores

        limpar_memoria_device(
            device
        )

    if not embeddings:
        raise RuntimeError(
            "Nenhum embedding OpenCLIP foi gerado."
        )

    embeddings = np.concatenate(
        embeddings,
        axis=0,
    )

    embeddings = normalizar_embeddings(
        embeddings
    )

    if (
        len(embeddings)
        != len(registros_processados)
    ):
        raise RuntimeError(
            "A quantidade de embeddings não corresponde "
            "à quantidade de registros processados."
        )

    return (
        embeddings,
        registros_processados,
    )


def gerar_embeddings(
    registros,
    device,
    batch_size,
    model_name,
    pretrained,
):
    """
    Carrega o OpenCLIP e gera os embeddings das imagens.
    """
    print("\nCarregando OpenCLIP")
    print(f"  Arquitetura: {model_name}")
    print(f"  Checkpoint: {pretrained}")

    try:
        model, _, preprocess = (
            open_clip.create_model_and_transforms(
                model_name=model_name,
                pretrained=pretrained,
            )
        )

    except Exception as erro:
        raise RuntimeError(
            "Não foi possível carregar o modelo OpenCLIP. "
            "Verifique a conexão com a internet, o nome do modelo "
            "e o checkpoint solicitado. "
            f"Erro original: {erro}"
        ) from erro

    model = model.to(
        device
    )

    model.eval()

    try:
        return extrair_embeddings(
            registros=registros,
            preprocess=preprocess,
            model=model,
            device=device,
            batch_size=batch_size,
        )

    finally:
        del model
        limpar_memoria_device(
            device
        )


# ============================================================
# MANIFESTO E REUTILIZAÇÃO DOS EMBEDDINGS
# ============================================================

def salvar_ordem_itens(
    caminho,
    registros,
    model_name,
    pretrained,
    embedding_dimension,
):
    """
    Salva a ordem exata dos itens correspondentes às linhas
    do arquivo .npy.
    """
    dados = {
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "embedding_library": "OpenCLIP",
        "model_name": model_name,
        "pretrained": pretrained,
        "embedding_dimension": (
            embedding_dimension
        ),
        "item_count": len(registros),
        "items": [
            {
                "id": registro.get(
                    "id"
                ),
                "image_path": registro.get(
                    "_image_path"
                ),
            }
            for registro in registros
        ],
    }

    salvar_json(
        caminho,
        dados,
    )


def reutilizar_embeddings(
    embeddings_path,
    items_path,
    registros_dashboard,
    model_name,
    pretrained,
):
    """
    Reutiliza embeddings existentes somente quando o manifesto
    confirma que foram gerados pelo mesmo modelo OpenCLIP.
    """
    if not embeddings_path.exists():
        raise FileNotFoundError(
            "Embeddings não encontrados: "
            f"{embeddings_path}"
        )

    if not items_path.exists():
        raise FileNotFoundError(
            "Manifesto dos embeddings não encontrado: "
            f"{items_path}"
        )

    embeddings = np.load(
        embeddings_path,
        allow_pickle=False,
    )

    manifesto = carregar_json(
        items_path
    )

    if isinstance(
        manifesto,
        list,
    ):
        raise ValueError(
            "O manifesto está no formato antigo, possivelmente "
            "criado para embeddings DINOv2. Execute novamente "
            "sem --reuse-embeddings."
        )

    if not isinstance(
        manifesto,
        dict,
    ):
        raise ValueError(
            "Formato inválido no manifesto dos embeddings."
        )

    embedding_library = manifesto.get(
        "embedding_library"
    )

    saved_model = manifesto.get(
        "model_name"
    )

    saved_pretrained = manifesto.get(
        "pretrained"
    )

    if embedding_library != "OpenCLIP":
        raise ValueError(
            "Os embeddings existentes não foram gerados "
            "com OpenCLIP. Execute sem --reuse-embeddings."
        )

    if saved_model != model_name:
        raise ValueError(
            "O modelo dos embeddings existentes não corresponde "
            "ao modelo solicitado.\n"
            f"Salvo: {saved_model}\n"
            f"Solicitado: {model_name}"
        )

    if saved_pretrained != pretrained:
        raise ValueError(
            "O checkpoint dos embeddings existentes não "
            "corresponde ao checkpoint solicitado.\n"
            f"Salvo: {saved_pretrained}\n"
            f"Solicitado: {pretrained}"
        )

    ordem_salva = manifesto.get(
        "items",
        [],
    )

    if not isinstance(
        ordem_salva,
        list,
    ):
        raise ValueError(
            "O campo 'items' do manifesto é inválido."
        )

    registros_por_id = {
        registro.get("id"): registro
        for registro in registros_dashboard
        if registro.get("id")
    }

    registros_ordenados = []

    for item in ordem_salva:
        item_id = item.get(
            "id"
        )

        registro = registros_por_id.get(
            item_id
        )

        if registro is None:
            raise ValueError(
                "Um item dos embeddings não existe mais "
                "no dashboard_data.json: "
                f"{item_id}"
            )

        registros_ordenados.append(
            registro
        )

    if embeddings.ndim != 2:
        raise ValueError(
            "O arquivo de embeddings deve ser uma matriz 2D."
        )

    if (
        len(embeddings)
        != len(registros_ordenados)
    ):
        raise ValueError(
            "A quantidade de embeddings não corresponde "
            "à quantidade de itens do manifesto."
        )

    saved_dimension = manifesto.get(
        "embedding_dimension"
    )

    if (
        saved_dimension is not None
        and int(saved_dimension)
        != embeddings.shape[1]
    ):
        raise ValueError(
            "A dimensão dos embeddings não corresponde "
            "à dimensão registrada no manifesto."
        )

    print(
        "Embeddings OpenCLIP reutilizados: "
        f"{embeddings.shape}"
    )

    return (
        normalizar_embeddings(
            embeddings
        ),
        registros_ordenados,
    )


# ============================================================
# PCA
# ============================================================

def reduzir_com_pca(
    embeddings,
    components,
    seed,
):
    """
    Reduz a dimensionalidade antes do HDBSCAN.

    A redução não é usada diretamente como coordenada visual.
    O UMAP é executado depois e cria a projeção em duas dimensões.
    """
    if components <= 0:
        print("PCA desativado")
        return embeddings, None

    max_components = min(
        embeddings.shape[0] - 1,
        embeddings.shape[1],
    )

    components = min(
        components,
        max_components,
    )

    if components < 2:
        print(
            "Poucos dados para PCA; "
            "usando embeddings originais."
        )

        return embeddings, None

    print(
        "Aplicando PCA: "
        f"{embeddings.shape[1]} -> "
        f"{components} dimensões"
    )

    pca = PCA(
        n_components=components,
        random_state=seed,
    )

    reduced = pca.fit_transform(
        embeddings
    ).astype(np.float32)

    reduced = normalizar_embeddings(
        reduced
    )

    info = {
        "components": components,
        "explained_variance_ratio": round(
            float(
                pca.explained_variance_ratio_.sum()
            ),
            6,
        ),
    }

    return reduced, info


# ============================================================
# UMAP
# ============================================================

def criar_projecao_umap(
    embeddings,
    neighbors,
    min_dist,
    seed,
):
    """
    Cria as coordenadas bidimensionais usadas no mapa interativo.
    """
    quantidade = len(
        embeddings
    )

    if quantidade < 3:
        raise ValueError(
            "São necessárias pelo menos três imagens "
            "para executar o UMAP."
        )

    numero_vizinhos = min(
        neighbors,
        quantidade - 1,
    )

    numero_vizinhos = max(
        2,
        numero_vizinhos,
    )

    print(
        "Executando UMAP "
        f"(n_neighbors={numero_vizinhos}, "
        f"min_dist={min_dist})"
    )

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=numero_vizinhos,
        min_dist=min_dist,
        metric="cosine",
        random_state=seed,
        low_memory=True,
    )

    coordinates = reducer.fit_transform(
        embeddings
    )

    return coordinates.astype(
        np.float32
    )


# ============================================================
# HDBSCAN
# ============================================================

def criar_clusters(
    embeddings,
    min_cluster_size,
    min_samples,
):
    """
    Executa HDBSCAN no espaço de embeddings, ou no espaço
    reduzido pelo PCA quando ele estiver habilitado.
    """
    quantidade = len(
        embeddings
    )

    if quantidade < 3:
        raise ValueError(
            "São necessárias pelo menos três imagens "
            "para executar o HDBSCAN."
        )

    tamanho_minimo = min(
        min_cluster_size,
        quantidade,
    )

    tamanho_minimo = max(
        2,
        tamanho_minimo,
    )

    amostras_minimas = min(
        min_samples,
        quantidade - 1,
    )

    amostras_minimas = max(
        1,
        amostras_minimas,
    )

    print(
        "Executando HDBSCAN "
        f"(min_cluster_size={tamanho_minimo}, "
        f"min_samples={amostras_minimas})"
    )

    agrupador = hdbscan.HDBSCAN(
        min_cluster_size=tamanho_minimo,
        min_samples=amostras_minimas,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    labels = agrupador.fit_predict(
        embeddings
    )

    probabilities = getattr(
        agrupador,
        "probabilities_",
        np.ones(
            len(labels),
            dtype=np.float32,
        ),
    )

    outlier_scores = getattr(
        agrupador,
        "outlier_scores_",
        np.zeros(
            len(labels),
            dtype=np.float32,
        ),
    )

    probabilities = np.nan_to_num(
        probabilities,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    outlier_scores = np.nan_to_num(
        outlier_scores,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    return (
        labels,
        probabilities,
        outlier_scores,
    )


# ============================================================
# ANÁLISE DOS CLUSTERS
# ============================================================

def valor_dominante(
    registros,
    campo,
):
    """
    Retorna o valor mais frequente de um campo.
    """
    valores = [
        str(
            registro.get(
                campo,
                "Unknown",
            )
        )
        for registro in registros
    ]

    if not valores:
        return {
            "value": "Unknown",
            "count": 0,
            "proportion": 0,
        }

    contador = Counter(
        valores
    )

    valor, quantidade = (
        contador.most_common(1)[0]
    )

    return {
        "value": valor,
        "count": quantidade,
        "proportion": round(
            quantidade / len(valores),
            6,
        ),
    }


def calcular_distribuicao(
    registros,
    campo,
):
    """
    Calcula a distribuição de um campo dentro de um cluster.
    """
    contador = Counter(
        str(
            registro.get(
                campo,
                "Unknown",
            )
        )
        for registro in registros
    )

    total = len(
        registros
    )

    return [
        {
            "value": valor,
            "count": quantidade,
            "proportion": round(
                quantidade / total,
                6,
            ) if total else 0,
        }
        for valor, quantidade
        in contador.most_common()
    ]


def calcular_resumo_clusters(
    labels,
    embeddings,
    registros,
):
    """
    Calcula estatísticas gerais e por cluster.
    """
    labels = np.asarray(
        labels
    )

    cluster_ids = sorted({
        int(label)
        for label in labels
        if label >= 0
    })

    noise_count = int(
        np.sum(
            labels == -1
        )
    )

    grouped_count = int(
        np.sum(
            labels >= 0
        )
    )

    silhouette = None

    mascara = (
        labels >= 0
    )

    labels_sem_ruido = (
        labels[mascara]
    )

    clusters_sem_ruido = set(
        labels_sem_ruido.tolist()
    )

    # O silhouette exige pelo menos dois clusters e menos
    # clusters do que amostras.
    if (
        len(labels_sem_ruido) >= 3
        and len(clusters_sem_ruido) >= 2
        and len(clusters_sem_ruido)
        < len(labels_sem_ruido)
    ):
        try:
            silhouette = float(
                silhouette_score(
                    embeddings[mascara],
                    labels_sem_ruido,
                    metric="cosine",
                )
            )

        except Exception as erro:
            print(
                "Warning: silhouette score não pôde "
                f"ser calculado: {erro}"
            )

            silhouette = None

    registros_por_cluster = defaultdict(
        list
    )

    for label, registro in zip(
        labels,
        registros,
    ):
        registros_por_cluster[
            int(label)
        ].append(registro)

    cluster_summaries = []

    for cluster_id in cluster_ids:
        itens = registros_por_cluster[
            cluster_id
        ]

        confiancas = [
            numero_finito(
                item.get(
                    "confidence"
                ),
                0,
            )
            for item in itens
        ]

        cluster_summaries.append({
            "cluster": cluster_id,
            "label": (
                f"Cluster {cluster_id}"
            ),
            "size": len(itens),
            "dominant_class": (
                valor_dominante(
                    itens,
                    "class",
                )
            ),
            "dominant_year": (
                valor_dominante(
                    itens,
                    "year",
                )
            ),
            "mean_yolo_confidence": (
                round(
                    sum(confiancas)
                    / len(confiancas),
                    6,
                )
                if confiancas
                else 0
            ),
            "class_distribution": (
                calcular_distribuicao(
                    itens,
                    "class",
                )
            ),
            "year_distribution": (
                calcular_distribuicao(
                    itens,
                    "year",
                )
            ),
        })

    cluster_summaries.sort(
        key=lambda item: (
            -item["size"],
            item["cluster"],
        )
    )

    total = len(
        labels
    )

    return {
        "items": total,
        "cluster_count": len(
            cluster_ids
        ),
        "grouped_items": grouped_count,
        "noise_items": noise_count,
        "noise_percent": (
            round(
                noise_count
                / total
                * 100,
                6,
            )
            if total
            else 0
        ),
        "silhouette_score": (
            round(
                silhouette,
                6,
            )
            if silhouette is not None
            else None
        ),
        "clusters": cluster_summaries,
    }


# ============================================================
# SAÍDA DOS ITENS
# ============================================================

def criar_itens_saida(
    registros,
    coordinates,
    labels,
    probabilities,
    outlier_scores,
):
    """
    Cria a estrutura compacta consumida pelo clusters.js.
    """
    items = []

    for indice, registro in enumerate(
        registros
    ):
        label = int(
            labels[indice]
        )

        item = {
            "id": registro.get(
                "id"
            ),
            "page_id": registro.get(
                "page_id"
            ),
            "x": round(
                float(
                    coordinates[
                        indice,
                        0,
                    ]
                ),
                6,
            ),
            "y": round(
                float(
                    coordinates[
                        indice,
                        1,
                    ]
                ),
                6,
            ),
            "cluster": label,
            "cluster_label": (
                "Noise / Unassigned"
                if label == -1
                else f"Cluster {label}"
            ),
            "cluster_probability": round(
                numero_finito(
                    probabilities[indice],
                    0,
                ),
                6,
            ),
            "outlier_score": round(
                numero_finito(
                    outlier_scores[indice],
                    0,
                ),
                6,
            ),
            "class": (
                registro.get("class")
                or "Unknown"
            ),
            "class_id": registro.get(
                "class_id"
            ),
            "confidence": round(
                numero_finito(
                    registro.get(
                        "confidence"
                    ),
                    0,
                ),
                6,
            ),
            "year": (
                registro.get("year")
                or "Unknown"
            ),
            "journal": (
                registro.get("journal")
                or "Unknown"
            ),
            "page": registro.get(
                "page"
            ),
            "issue": registro.get(
                "issue"
            ),
            "bib": registro.get(
                "bib"
            ),
            "bbox": registro.get(
                "bbox"
            ),
            "crop_image": registro.get(
                "crop_image"
            ),
            "page_image": registro.get(
                "page_image"
            ),
            "url": registro.get(
                "url"
            ),
        }

        items.append(
            item
        )

    return items


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    configurar_sementes(
        args.seed
    )

    dashboard_path = (
        args.dashboard_data
        .expanduser()
        .resolve()
    )

    output_path = (
        args.output
        .expanduser()
        .resolve()
    )

    embeddings_path = (
        args.embeddings_output
        .expanduser()
        .resolve()
    )

    items_path = (
        args.items_output
        .expanduser()
        .resolve()
    )

    dashboard = carregar_json(
        dashboard_path
    )

    registros_dashboard = preparar_registros(
        dashboard=dashboard,
        dashboard_directory=(
            dashboard_path.parent
        ),
        max_items=args.max_items,
    )

    if len(registros_dashboard) < 3:
        raise ValueError(
            "Não há imagens suficientes para clustering. "
            "São necessárias pelo menos três imagens."
        )

    device = detectar_device()

    print("\n" + "=" * 70)
    print("VISUAL CLUSTERING COM OPENCLIP")
    print("=" * 70)
    print(
        f"Dashboard: {dashboard_path}"
    )
    print(
        f"Modelo OpenCLIP: {args.model_name}"
    )
    print(
        f"Checkpoint: {args.pretrained}"
    )
    print(
        f"Device: {device}"
    )
    print(
        "Imagens disponíveis: "
        f"{len(registros_dashboard)}"
    )
    print(
        "Máximo solicitado: "
        f"{args.max_items or 'todos'}"
    )
    print(
        "Reutilizar embeddings: "
        f"{args.reuse_embeddings}"
    )
    print("=" * 70)

    # ========================================================
    # EMBEDDINGS
    # ========================================================

    if args.reuse_embeddings:
        (
            embeddings,
            registros_processados,
        ) = reutilizar_embeddings(
            embeddings_path=embeddings_path,
            items_path=items_path,
            registros_dashboard=(
                registros_dashboard
            ),
            model_name=args.model_name,
            pretrained=args.pretrained,
        )

    else:
        (
            embeddings,
            registros_processados,
        ) = gerar_embeddings(
            registros=(
                registros_dashboard
            ),
            device=device,
            batch_size=args.batch_size,
            model_name=args.model_name,
            pretrained=args.pretrained,
        )

        embeddings_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.save(
            embeddings_path,
            embeddings,
            allow_pickle=False,
        )

        salvar_ordem_itens(
            caminho=items_path,
            registros=registros_processados,
            model_name=args.model_name,
            pretrained=args.pretrained,
            embedding_dimension=(
                embeddings.shape[1]
            ),
        )

    print(
        "Formato dos embeddings: "
        f"{embeddings.shape}"
    )

    # ========================================================
    # PCA
    # ========================================================

    embeddings_cluster, pca_info = (
        reduzir_com_pca(
            embeddings=embeddings,
            components=(
                args.pca_components
            ),
            seed=args.seed,
        )
    )

    # ========================================================
    # UMAP
    # ========================================================

    coordinates = criar_projecao_umap(
        embeddings=embeddings_cluster,
        neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        seed=args.seed,
    )

    # ========================================================
    # HDBSCAN
    # ========================================================

    (
        labels,
        probabilities,
        outlier_scores,
    ) = criar_clusters(
        embeddings=embeddings_cluster,
        min_cluster_size=(
            args.min_cluster_size
        ),
        min_samples=args.min_samples,
    )

    # ========================================================
    # RESUMO
    # ========================================================

    summary = calcular_resumo_clusters(
        labels=labels,
        embeddings=embeddings_cluster,
        registros=registros_processados,
    )

    items = criar_itens_saida(
        registros=registros_processados,
        coordinates=coordinates,
        labels=labels,
        probabilities=probabilities,
        outlier_scores=outlier_scores,
    )

    numero_vizinhos_real = min(
        args.umap_neighbors,
        len(registros_processados) - 1,
    )

    numero_vizinhos_real = max(
        2,
        numero_vizinhos_real,
    )

    # ========================================================
    # JSON FINAL
    # ========================================================

    resultado = {
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "method": {
            "embedding_library": (
                "OpenCLIP"
            ),
            "embedding_model": (
                args.model_name
            ),
            "embedding_pretrained": (
                args.pretrained
            ),
            "embedding_dimension": int(
                embeddings.shape[1]
            ),
            "embedding_normalization": (
                "L2"
            ),
            "pca": pca_info,
            "dimensionality_reduction": (
                "UMAP"
            ),
            "umap_components": 2,
            "umap_neighbors": (
                numero_vizinhos_real
            ),
            "umap_min_dist": (
                args.umap_min_dist
            ),
            "umap_metric": "cosine",
            "clustering": "HDBSCAN",
            "clustering_space": (
                "PCA-reduced normalized "
                "OpenCLIP image embeddings"
                if pca_info is not None
                else (
                    "normalized OpenCLIP "
                    "image embeddings"
                )
            ),
            "hdbscan_metric": (
                "euclidean"
            ),
            "min_cluster_size": (
                args.min_cluster_size
            ),
            "min_samples": (
                args.min_samples
            ),
            "random_state": (
                args.seed
            ),
        },
        "summary": summary,
        "items": items,
    }

    salvar_json(
        output_path,
        resultado,
    )

    # ========================================================
    # RESUMO FINAL
    # ========================================================

    print("\n" + "=" * 70)
    print("CLUSTERING COMPLETED")
    print("=" * 70)
    print(
        "Biblioteca de embeddings:    "
        "OpenCLIP"
    )
    print(
        "Modelo:                      "
        f"{args.model_name}"
    )
    print(
        "Checkpoint:                  "
        f"{args.pretrained}"
    )
    print(
        "Imagens processadas:         "
        f"{len(registros_processados)}"
    )
    print(
        "Dimensão dos embeddings:     "
        f"{embeddings.shape[1]}"
    )
    print(
        "Dimensão para clustering:    "
        f"{embeddings_cluster.shape[1]}"
    )
    print(
        "Clusters encontrados:        "
        f"{summary['cluster_count']}"
    )
    print(
        "Imagens agrupadas:           "
        f"{summary['grouped_items']}"
    )
    print(
        "Imagens como ruído:          "
        f"{summary['noise_items']}"
    )
    print(
        "Percentual de ruído:         "
        f"{summary['noise_percent']:.2f}%"
    )
    print(
        "Silhouette score:            "
        f"{summary['silhouette_score']}"
    )
    print()
    print(
        f"JSON para o dashboard: {output_path}"
    )
    print(
        f"Embeddings locais: {embeddings_path}"
    )
    print(
        f"Manifesto dos itens: {items_path}"
    )
    print("=" * 70)

    if summary["cluster_count"] == 0:
        print(
            "\nAVISO: nenhum cluster foi encontrado."
        )
        print(
            "Tente parâmetros menos restritivos:"
        )
        print(
            "  --min-cluster-size 8 --min-samples 3"
        )

    elif summary["noise_percent"] > 70:
        print(
            "\nAVISO: mais de 70% das imagens foram "
            "classificadas como ruído."
        )
        print(
            "Considere reduzir --min-cluster-size "
            "ou --min-samples."
        )

    elif summary["cluster_count"] > 50:
        print(
            "\nAVISO: foi encontrado um número muito alto "
            "de clusters."
        )
        print(
            "Considere aumentar --min-cluster-size "
            "ou --min-samples."
        )


if __name__ == "__main__":
    main()