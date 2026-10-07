"""
Prepara os dados e os recursos visuais do dashboard Quarto.

Este script:

    - lê os resultados completos da inferência;
    - lê o registro de todas as páginas processadas;
    - lê os metadados originais;
    - normaliza as bounding boxes para o heatmap espacial;
    - gera thumbnails WebP para a galeria;
    - gera previews WebP das páginas;
    - seleciona uma amostra para a galeria e o clustering;
    - copia os diagnósticos do treinamento;
    - lê o histórico de treinamento do Ultralytics;
    - gera dashboard_data.json.

Importante:

    - O heatmap usa TODAS as detecções válidas.
    - A galeria usa uma amostra limitada.
    - O clustering posterior usa os mesmos itens da galeria.
    - Os PNGs originais não são copiados para o site.

Exemplo:

python3 prepare_dashboard.py \
  --detections "/Users/luanamoraescosta/DH/ilustracao/ilustracaobrasil/resultados_inferencia_resultados.json" \
  --pages "/Users/luanamoraescosta/DH/ilustracao/ilustracaobrasil/resultados_inferencia_paginas.json" \
  --metadata "/Users/luanamoraescosta/DH/ilustracao/ilustracaobrasil/metadata_enriquecido.json" \
  --inference-manifest "/Users/luanamoraescosta/DH/ilustracao/ilustracaobrasil/resultados_inferencia_execucao.json" \
  --training-dir "/Users/luanamoraescosta/DH/ilustracao/finetune/runs/detect/runs_jornal/layout_jornal_det" \
  --max-gallery-items 5000 \
  --sampling stratified \
  --seed 42
"""

import argparse
import hashlib
import json
import math
import random
import shutil
from collections import Counter, defaultdict, deque
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm


# ============================================================
# CONFIGURAÇÃO
# ============================================================

DIAGNOSTIC_FILES = [
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
    "PR_curve.png",
    "F1_curve.png",
    "P_curve.png",
    "R_curve.png",
    "results.png",
    "labels.jpg",
    "labels_correlogram.jpg",
    "val_batch0_labels.jpg",
    "val_batch0_pred.jpg",
    "val_batch1_labels.jpg",
    "val_batch1_pred.jpg",
    "val_batch2_labels.jpg",
    "val_batch2_pred.jpg",
]

# Tamanho dos recortes exibidos na galeria e no Cluster Atlas.
CROP_THUMBNAIL_SIZE = (600, 600)
CROP_WEBP_QUALITY = 76

# Tamanho dos previews das páginas exibidos nos modais.
PAGE_THUMBNAIL_SIZE = (800, 1200)
PAGE_WEBP_QUALITY = 65


# ============================================================
# ARGUMENTOS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Prepara dados estáticos e imagens otimizadas "
            "para o dashboard Quarto."
        )
    )

    parser.add_argument(
        "--detections",
        type=Path,
        required=True,
        help=(
            "JSON com um registro para cada detecção."
        ),
    )

    parser.add_argument(
        "--pages",
        type=Path,
        required=True,
        help=(
            "JSON com um registro para cada página processada."
        ),
    )

    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help=(
            "Arquivo original de metadados."
        ),
    )

    parser.add_argument(
        "--inference-manifest",
        type=Path,
        default=None,
        help=(
            "JSON com parâmetros e resumo da inferência."
        ),
    )

    parser.add_argument(
        "--training-dir",
        type=Path,
        required=True,
        help=(
            "Diretório do treinamento YOLO que contém "
            "results.csv, args.yaml e os diagnósticos."
        ),
    )

    parser.add_argument(
        "--validation-dir",
        type=Path,
        default=None,
        help=(
            "Diretório opcional com os resultados da validação. "
            "Se omitido, procura training-dir/validation."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dashboard_data.json"),
        help=(
            "Caminho do JSON final utilizado pelo dashboard."
        ),
    )

    parser.add_argument(
        "--assets",
        type=Path,
        default=Path("dashboard_assets"),
        help=(
            "Diretório dos recursos estáticos do dashboard."
        ),
    )

    parser.add_argument(
        "--max-gallery-items",
        type=int,
        default=1500,
        help=(
            "Número máximo de imagens na galeria e no clustering. "
            "Use 0 para incluir todas as detecções."
        ),
    )

    parser.add_argument(
        "--sampling",
        type=str,
        choices=[
            "stratified",
            "random",
            "confidence",
        ],
        default="stratified",
        help=(
            "Estratégia para selecionar as imagens da galeria. "
            "'stratified' equilibra ano e classe; "
            "'random' usa amostragem aleatória; "
            "'confidence' seleciona as maiores confianças."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Semente usada nas amostragens random e stratified."
        ),
    )

    parser.add_argument(
        "--keep-assets",
        action="store_true",
        help=(
            "Não remove dashboard_assets antes de executar. "
            "Por padrão, os assets anteriores são apagados."
        ),
    )

    args = parser.parse_args()

    if args.max_gallery_items < 0:
        parser.error(
            "--max-gallery-items deve ser maior ou igual a zero"
        )

    return args


# ============================================================
# JSON
# ============================================================

def carregar_json(caminho):
    caminho = Path(caminho).expanduser()

    if not caminho.exists():
        raise FileNotFoundError(
            f"Arquivo JSON não encontrado: {caminho}"
        )

    with caminho.open(
        "r",
        encoding="utf-8",
    ) as arquivo:
        return json.load(arquivo)


def salvar_json(caminho, dados):
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


# ============================================================
# CONVERSÃO E VALIDAÇÃO DE VALORES
# ============================================================

def valor_vazio(valor):
    if valor is None:
        return True

    if isinstance(valor, str):
        return valor.strip().lower() in {
            "",
            "none",
            "null",
            "nan",
            "unknown",
        }

    try:
        return bool(pd.isna(valor))
    except (TypeError, ValueError):
        return False


def numero(valor, padrao=None):
    try:
        resultado = float(valor)

        if not math.isfinite(resultado):
            return padrao

        return resultado

    except (TypeError, ValueError):
        return padrao


def inteiro(valor, padrao=None):
    try:
        resultado = float(valor)

        if not math.isfinite(resultado):
            return padrao

        return int(resultado)

    except (TypeError, ValueError):
        return padrao


def booleano(valor):
    if isinstance(valor, bool):
        return valor

    if isinstance(valor, (int, float)):
        return bool(valor)

    if isinstance(valor, str):
        return valor.strip().lower() in {
            "true",
            "1",
            "yes",
            "sim",
        }

    return False


def ano_normalizado(valor):
    ano = inteiro(valor)

    if ano is None:
        return "Unknown"

    return str(ano)


def caminho_valido(valor):
    """
    Retorna um Path somente se o valor indicar um arquivo existente.
    """
    if valor_vazio(valor):
        return None

    caminho = Path(
        str(valor)
    ).expanduser()

    if not caminho.exists():
        return None

    if not caminho.is_file():
        return None

    return caminho


# ============================================================
# IMAGENS
# ============================================================

def hash_texto(valor):
    return hashlib.sha1(
        str(valor).encode("utf-8")
    ).hexdigest()[:16]


def hash_caminho(caminho):
    if caminho is None:
        return hash_texto("missing")

    try:
        texto = str(
            Path(caminho)
            .expanduser()
            .resolve()
        )
    except Exception:
        texto = str(caminho)

    return hash_texto(texto)


def criar_thumbnail(
    origem,
    destino,
    max_size,
    quality=82,
):
    """
    Cria um thumbnail WebP sem ampliar imagens pequenas.

    Retorna True quando a imagem foi criada com sucesso.
    """
    if origem is None:
        return False

    origem = Path(origem).expanduser()
    destino = Path(destino)

    if not origem.exists() or not origem.is_file():
        return False

    destino.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        with Image.open(origem) as imagem:
            imagem = ImageOps.exif_transpose(
                imagem
            )

            # Converte transparência para fundo branco.
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

                imagem = fundo.convert("RGB")
            else:
                imagem = imagem.convert("RGB")

            imagem.thumbnail(
                max_size,
                Image.Resampling.LANCZOS,
            )

            imagem.save(
                destino,
                format="WEBP",
                quality=quality,
                method=6,
                optimize=True,
            )

        return True

    except Exception as erro:
        print(
            "Warning: thumbnail failed for "
            f"{origem}: {erro}"
        )

        return False


def copiar_se_existir(
    origem,
    destino,
):
    origem = Path(origem)

    if not origem.exists():
        return False

    if not origem.is_file():
        return False

    destino = Path(destino)

    destino.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        origem,
        destino,
    )

    return True


# ============================================================
# DIAGNÓSTICOS DO TREINAMENTO
# ============================================================

def encontrar_diagnostico(
    nome,
    training_dir,
    validation_dir,
):
    candidatos = []

    if validation_dir is not None:
        candidatos.append(
            validation_dir / nome
        )

    candidatos.append(
        training_dir / nome
    )

    # Algumas execuções antigas podem ter diretórios val, val2 etc.
    for diretorio in sorted(
        training_dir.glob("val*")
    ):
        if diretorio.is_dir():
            candidatos.append(
                diretorio / nome
            )

    for candidato in candidatos:
        if candidato.exists():
            return candidato

    return None


def carregar_manifesto(
    training_dir,
):
    candidatos = [
        (
            training_dir
            / "experiment_manifest.json"
        ),
        (
            training_dir
            / "resumo_metricas.json"
        ),
    ]

    for caminho in candidatos:
        if caminho.exists():
            return carregar_json(caminho)

    return {}


def carregar_historico_treinamento(
    training_dir,
    data_dir,
):
    results_csv = (
        training_dir
        / "results.csv"
    )

    if not results_csv.exists():
        print(
            "Warning: results.csv não encontrado em "
            f"{training_dir}"
        )

        return []

    training_df = pd.read_csv(
        results_csv
    )

    training_df.columns = [
        str(coluna).strip()
        for coluna in training_df.columns
    ]

    copiar_se_existir(
        results_csv,
        data_dir
        / "training_results.csv",
    )

    # A conversão via to_json remove tipos NumPy e converte NaN
    # em null, produzindo dados seguros para json.dumps.
    return json.loads(
        training_df.to_json(
            orient="records"
        )
    )


# ============================================================
# METADADOS
# ============================================================

def calcular_qualidade_metadata(
    metadata,
):
    total = len(metadata)

    campos = sorted({
        chave
        for item in metadata
        if isinstance(item, dict)
        for chave in item.keys()
    })

    qualidade = []

    for campo in campos:
        ausentes = sum(
            1
            for item in metadata
            if (
                not isinstance(item, dict)
                or campo not in item
                or valor_vazio(
                    item.get(campo)
                )
            )
        )

        qualidade.append({
            "field": campo,
            "missing": ausentes,
            "present": total - ausentes,
            "missing_percent": (
                round(
                    ausentes
                    / total
                    * 100,
                    6,
                )
                if total
                else 0
            ),
        })

    qualidade.sort(
        key=lambda item: (
            item["missing_percent"],
            item["field"],
        ),
        reverse=True,
    )

    return qualidade


# ============================================================
# BOUNDING BOXES E HEATMAP
# ============================================================

def normalizar_bbox(
    bbox,
    largura,
    altura,
):
    """
    Converte uma bounding box de pixels para coordenadas de 0 a 1.

    Entrada:
        [x1, y1, x2, y2]

    Saída:
        [x1_normalizado, y1_normalizado,
         x2_normalizado, y2_normalizado]
    """
    if (
        not isinstance(bbox, (list, tuple))
        or len(bbox) != 4
        or largura is None
        or altura is None
        or largura <= 0
        or altura <= 0
    ):
        return None

    coordenadas = [
        numero(valor)
        for valor in bbox
    ]

    if any(
        valor is None
        for valor in coordenadas
    ):
        return None

    x1, y1, x2, y2 = coordenadas

    x1 = max(
        0.0,
        min(
            1.0,
            x1 / largura,
        ),
    )

    y1 = max(
        0.0,
        min(
            1.0,
            y1 / altura,
        ),
    )

    x2 = max(
        0.0,
        min(
            1.0,
            x2 / largura,
        ),
    )

    y2 = max(
        0.0,
        min(
            1.0,
            y2 / altura,
        ),
    )

    if x2 <= x1 or y2 <= y1:
        return None

    return [
        round(x1, 6),
        round(y1, 6),
        round(x2, 6),
        round(y2, 6),
    ]


# ============================================================
# AMOSTRAGEM PARA GALERIA E CLUSTERING
# ============================================================

def selecionar_deteccoes(
    detections,
    limite,
    estrategia="stratified",
    seed=42,
):
    """
    Seleciona detecções para a galeria e para o clustering.

    Estratégias:

        confidence:
            seleciona primeiro as maiores confianças;

        random:
            gera uma amostra aleatória reproduzível;

        stratified:
            intercala grupos formados por ano e classe YOLO,
            reduzindo a dominância de anos/classes frequentes.

    Um limite igual a zero utiliza todas as detecções.
    """
    registros = [
        item
        for item in detections
        if isinstance(item, dict)
    ]

    if limite <= 0 or limite >= len(registros):
        return registros

    gerador = random.Random(seed)

    if estrategia == "confidence":
        return sorted(
            registros,
            key=lambda item: numero(
                item.get("confianca"),
                0,
            ),
            reverse=True,
        )[:limite]

    if estrategia == "random":
        gerador.shuffle(registros)
        return registros[:limite]

    # ========================================================
    # AMOSTRAGEM ESTRATIFICADA POR ANO E CLASSE
    # ========================================================

    grupos = defaultdict(list)

    for registro in registros:
        ano = ano_normalizado(
            registro.get("ano")
        )

        classe = (
            registro.get("nome_classe")
            or "Unknown"
        )

        grupos[
            (
                ano,
                str(classe),
            )
        ].append(registro)

    filas = {}

    for chave, itens in grupos.items():
        gerador.shuffle(itens)

        filas[chave] = deque(
            itens
        )

    chaves = list(
        filas.keys()
    )

    gerador.shuffle(chaves)

    selecionados = []

    # Seleção round-robin:
    # retira um item de cada estrato por rodada.
    while (
        len(selecionados) < limite
        and chaves
    ):
        chaves_ativas = []

        for chave in chaves:
            fila = filas[chave]

            if fila:
                selecionados.append(
                    fila.popleft()
                )

            if fila:
                chaves_ativas.append(
                    chave
                )

            if len(selecionados) >= limite:
                break

        chaves = chaves_ativas

    return selecionados


# ============================================================
# PREPARAÇÃO DAS PÁGINAS
# ============================================================

def preparar_paginas(
    pages,
):
    compact_pages = []

    for page in pages:
        if not isinstance(page, dict):
            continue

        page_id = page.get(
            "page_id"
        )

        if not page_id:
            continue

        compact_pages.append({
            "page_id": page_id,
            "year": ano_normalizado(
                page.get("ano")
            ),
            "journal": (
                page.get("revista")
                or "Unknown"
            ),
            "page": page.get(
                "pagina"
            ),
            "issue": page.get(
                "num"
            ),
            "bib": page.get(
                "bib"
            ),
            "url": page.get(
                "url"
            ),
            "width": inteiro(
                page.get(
                    "largura_pagina"
                )
            ),
            "height": inteiro(
                page.get(
                    "altura_pagina"
                )
            ),
            "detection_count": inteiro(
                page.get(
                    "quantidade_deteccoes"
                ),
                0,
            ),
            "has_detection": booleano(
                page.get(
                    "tem_deteccao"
                )
            ),
            "status": (
                page.get("status")
                or "processed"
            ),
        })

    return compact_pages


# ============================================================
# DETECÇÕES ESPACIAIS
# ============================================================

def preparar_deteccoes_espaciais(
    detections,
):
    """
    Prepara dados leves para o heatmap.

    Nenhuma imagem é incluída nessa estrutura.
    Todas as detecções com bounding boxes válidas são preservadas.
    """
    spatial_detections = []

    for detection in detections:
        if not isinstance(
            detection,
            dict,
        ):
            continue

        largura = numero(
            detection.get(
                "largura_pagina"
            )
        )

        altura = numero(
            detection.get(
                "altura_pagina"
            )
        )

        bbox = normalizar_bbox(
            detection.get("bbox"),
            largura,
            altura,
        )

        if bbox is None:
            continue

        largura_bbox = (
            bbox[2] - bbox[0]
        )

        altura_bbox = (
            bbox[3] - bbox[1]
        )

        area_normalizada = (
            largura_bbox
            * altura_bbox
        )

        centro_x = (
            bbox[0] + bbox[2]
        ) / 2

        centro_y = (
            bbox[1] + bbox[3]
        ) / 2

        spatial_detections.append({
            "id": detection.get(
                "id"
            ),
            "page_id": detection.get(
                "page_id"
            ),
            "year": ano_normalizado(
                detection.get("ano")
            ),
            "class": (
                detection.get(
                    "nome_classe"
                )
                or "Unknown"
            ),
            "confidence": round(
                numero(
                    detection.get(
                        "confianca"
                    ),
                    0,
                ),
                6,
            ),
            "bbox": bbox,
            "center_x": round(
                centro_x,
                6,
            ),
            "center_y": round(
                centro_y,
                6,
            ),
            "normalized_area": round(
                area_normalizada,
                8,
            ),
        })

    return spatial_detections


# ============================================================
# GALERIA E THUMBNAILS
# ============================================================

def preparar_galeria(
    gallery_source,
    pages_by_id,
    crops_dir,
    pages_dir,
):
    gallery = []

    # Armazena o resultado da geração de cada preview para evitar
    # abrir e converter a mesma página várias vezes.
    preview_cache = {}

    print("\nGerando thumbnails da galeria...")

    barra = tqdm(
        gallery_source,
        total=len(gallery_source),
        desc="Thumbnails",
    )

    for detection in barra:
        detection_id = detection.get(
            "id"
        )

        if not detection_id:
            continue

        crop_original = caminho_valido(
            detection.get(
                "elemento_extraido"
            )
        )

        preview_original = caminho_valido(
            detection.get(
                "imagem_detectada"
            )
        )

        crop_name = (
            f"{detection_id}.webp"
        )

        preview_hash = hash_caminho(
            preview_original
            or detection.get(
                "imagem_detectada"
            )
            or detection.get(
                "page_id"
            )
        )

        preview_name = (
            f"{preview_hash}.webp"
        )

        crop_destination = (
            crops_dir
            / crop_name
        )

        preview_destination = (
            pages_dir
            / preview_name
        )

        crop_created = criar_thumbnail(
            origem=crop_original,
            destino=crop_destination,
            max_size=CROP_THUMBNAIL_SIZE,
            quality=CROP_WEBP_QUALITY,
        )

        preview_key = (
            str(preview_original)
            if preview_original is not None
            else str(
                detection.get(
                    "imagem_detectada"
                )
            )
        )

        if preview_key not in preview_cache:
            preview_cache[preview_key] = (
                criar_thumbnail(
                    origem=preview_original,
                    destino=(
                        preview_destination
                    ),
                    max_size=(
                        PAGE_THUMBNAIL_SIZE
                    ),
                    quality=(
                        PAGE_WEBP_QUALITY
                    ),
                )
            )

        page = pages_by_id.get(
            detection.get(
                "page_id"
            ),
            {},
        )

        largura_pagina = inteiro(
            detection.get(
                "largura_pagina"
            )
        )

        altura_pagina = inteiro(
            detection.get(
                "altura_pagina"
            )
        )

        bbox_normalizada = normalizar_bbox(
            detection.get("bbox"),
            largura_pagina,
            altura_pagina,
        )

        gallery.append({
            "id": detection_id,
            "page_id": detection.get(
                "page_id"
            ),
            "class": (
                detection.get(
                    "nome_classe"
                )
                or "Unknown"
            ),
            "class_id": inteiro(
                detection.get(
                    "classe"
                )
            ),
            "confidence": round(
                numero(
                    detection.get(
                        "confianca"
                    ),
                    0,
                ),
                6,
            ),
            "journal": (
                detection.get(
                    "revista"
                )
                or "Unknown"
            ),
            "year": ano_normalizado(
                detection.get(
                    "ano"
                )
            ),
            "page": detection.get(
                "pagina"
            ),
            "issue": detection.get(
                "num"
            ),
            "bib": detection.get(
                "bib"
            ),
            "url": detection.get(
                "url"
            ),
            "bbox": detection.get(
                "bbox"
            ),
            "normalized_bbox": (
                bbox_normalizada
            ),
            "bbox_crop": detection.get(
                "bbox_recorte"
            ),
            "crop_width": inteiro(
                detection.get(
                    "largura_recorte"
                )
            ),
            "crop_height": inteiro(
                detection.get(
                    "altura_recorte"
                )
            ),
            "page_width": (
                largura_pagina
            ),
            "page_height": (
                altura_pagina
            ),
            "bbox_area_percent": numero(
                detection.get(
                    "area_bbox_percentual_pagina"
                )
            ),
            "crop_area_percent": numero(
                detection.get(
                    "area_recorte_percentual_pagina"
                )
            ),
            "detections_on_page": inteiro(
                page.get(
                    "quantidade_deteccoes"
                ),
                inteiro(
                    page.get(
                        "detection_count"
                    ),
                    0,
                ),
            ),
            "crop_image": (
                "dashboard_assets/"
                f"crops/{crop_name}"
                if crop_created
                else None
            ),
            "page_image": (
                "dashboard_assets/"
                f"pages/{preview_name}"
                if preview_cache.get(
                    preview_key
                )
                else None
            ),
        })

    return gallery


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    # Resolve os caminhos de entrada.
    detections_path = (
        args.detections
        .expanduser()
        .resolve()
    )

    pages_path = (
        args.pages
        .expanduser()
        .resolve()
    )

    metadata_path = (
        args.metadata
        .expanduser()
        .resolve()
    )

    training_dir = (
        args.training_dir
        .expanduser()
        .resolve()
    )

    if not training_dir.exists():
        raise FileNotFoundError(
            "Diretório de treinamento não encontrado: "
            f"{training_dir}"
        )

    validation_dir = (
        args.validation_dir
    )

    if validation_dir is None:
        candidato = (
            training_dir
            / "validation"
        )

        if candidato.exists():
            validation_dir = candidato
        else:
            validation_dir = None

    else:
        validation_dir = (
            validation_dir
            .expanduser()
            .resolve()
        )

        if not validation_dir.exists():
            print(
                "Warning: diretório de validação "
                f"não encontrado: {validation_dir}"
            )

            validation_dir = None

    output_path = (
        args.output
        .expanduser()
        .resolve()
    )

    assets = (
        args.assets
        .expanduser()
        .resolve()
    )

    print("=" * 70)
    print("PREPARAÇÃO DO DASHBOARD")
    print("=" * 70)
    print(
        f"Detecções: {detections_path}"
    )
    print(
        f"Páginas: {pages_path}"
    )
    print(
        f"Metadados: {metadata_path}"
    )
    print(
        f"Treinamento: {training_dir}"
    )
    print(
        "Validação: "
        f"{validation_dir or 'não informada'}"
    )
    print(
        f"Saída JSON: {output_path}"
    )
    print(
        f"Assets: {assets}"
    )
    print(
        "Máximo da galeria: "
        f"{args.max_gallery_items or 'todos'}"
    )
    print(
        f"Amostragem: {args.sampling}"
    )
    print(
        f"Semente: {args.seed}"
    )
    print("=" * 70)

    # ========================================================
    # CARREGAMENTO DOS DADOS
    # ========================================================

    detections = carregar_json(
        detections_path
    )

    pages = carregar_json(
        pages_path
    )

    metadata = carregar_json(
        metadata_path
    )

    if not isinstance(
        detections,
        list,
    ):
        raise ValueError(
            "O JSON de detecções precisa conter uma lista."
        )

    if not isinstance(
        pages,
        list,
    ):
        raise ValueError(
            "O JSON de páginas precisa conter uma lista."
        )

    if not isinstance(
        metadata,
        list,
    ):
        raise ValueError(
            "O JSON de metadados precisa conter uma lista."
        )

    # ========================================================
    # DIRETÓRIOS DE SAÍDA
    # ========================================================

    crops_dir = (
        assets
        / "crops"
    )

    pages_dir = (
        assets
        / "pages"
    )

    diagnostics_dir = (
        assets
        / "diagnostics"
    )

    data_dir = (
        assets
        / "data"
    )

    if (
        assets.exists()
        and not args.keep_assets
    ):
        print(
            "\nRemovendo assets anteriores: "
            f"{assets}"
        )

        shutil.rmtree(
            assets
        )

    crops_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pages_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    diagnostics_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    data_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # PÁGINAS
    # ========================================================

    compact_pages = preparar_paginas(
        pages
    )

    pages_by_id = {
        item.get("page_id"): item
        for item in pages
        if (
            isinstance(item, dict)
            and item.get("page_id")
        )
    }

    # ========================================================
    # DETECÇÕES PARA O HEATMAP
    # ========================================================

    spatial_detections = (
        preparar_deteccoes_espaciais(
            detections
        )
    )

    # ========================================================
    # AMOSTRA PARA GALERIA E CLUSTERS
    # ========================================================

    gallery_source = selecionar_deteccoes(
        detections=detections,
        limite=args.max_gallery_items,
        estrategia=args.sampling,
        seed=args.seed,
    )

    print(
        "\nGallery sampling: "
        f"{args.sampling}"
    )

    print(
        "Selected gallery items: "
        f"{len(gallery_source)} "
        f"of {len(detections)} detections"
    )

    gallery = preparar_galeria(
        gallery_source=gallery_source,
        pages_by_id=pages_by_id,
        crops_dir=crops_dir,
        pages_dir=pages_dir,
    )

    gallery_with_images = sum(
        1
        for item in gallery
        if item.get("crop_image")
    )

    # ========================================================
    # QUALIDADE E COBERTURA DOS METADADOS
    # ========================================================

    metadata_quality = (
        calcular_qualidade_metadata(
            metadata
        )
    )

    metadata_years = Counter(
        ano_normalizado(
            item.get("ano")
        )
        for item in metadata
        if isinstance(item, dict)
    )

    page_years = Counter(
        item["year"]
        for item in compact_pages
    )

    detection_classes = Counter(
        item["class"]
        for item in spatial_detections
    )

    # ========================================================
    # HISTÓRICO E MANIFESTO DO TREINAMENTO
    # ========================================================

    training_results = (
        carregar_historico_treinamento(
            training_dir=training_dir,
            data_dir=data_dir,
        )
    )

    training_manifest = (
        carregar_manifesto(
            training_dir
        )
    )

    # ========================================================
    # MANIFESTO DA INFERÊNCIA
    # ========================================================

    inference_manifest = {}

    if args.inference_manifest is not None:
        inference_manifest_path = (
            args.inference_manifest
            .expanduser()
            .resolve()
        )

        if inference_manifest_path.exists():
            inference_manifest = carregar_json(
                inference_manifest_path
            )
        else:
            print(
                "Warning: manifesto da inferência "
                "não encontrado: "
                f"{inference_manifest_path}"
            )

    # ========================================================
    # DIAGNÓSTICOS DO TREINAMENTO
    # ========================================================

    copied_diagnostics = {}

    for filename in DIAGNOSTIC_FILES:
        source = encontrar_diagnostico(
            nome=filename,
            training_dir=training_dir,
            validation_dir=validation_dir,
        )

        if source is None:
            continue

        destination = (
            diagnostics_dir
            / filename
        )

        shutil.copy2(
            source,
            destination,
        )

        copied_diagnostics[filename] = (
            "dashboard_assets/"
            f"diagnostics/{filename}"
        )

    # ========================================================
    # RESUMOS DERIVADOS
    # ========================================================

    pages_with_detections = sum(
        1
        for item in compact_pages
        if item["has_detection"]
    )

    pages_without_detections = (
        len(compact_pages)
        - pages_with_detections
    )

    mean_confidence = (
        sum(
            item["confidence"]
            for item in spatial_detections
        )
        / len(spatial_detections)
        if spatial_detections
        else 0
    )

    mean_bbox_area = (
        sum(
            item["normalized_area"]
            for item in spatial_detections
        )
        / len(spatial_detections)
        if spatial_detections
        else 0
    )

    # ========================================================
    # JSON FINAL
    # ========================================================

    bundle = {
        "summary": {
            "metadata_records": len(
                metadata
            ),
            "processed_pages": len(
                compact_pages
            ),
            "pages_with_detections": (
                pages_with_detections
            ),
            "pages_without_detections": (
                pages_without_detections
            ),
            "detections": len(
                spatial_detections
            ),
            "available_gallery_detections": len(
                detections
            ),
            "gallery_items": len(
                gallery
            ),
            "gallery_items_with_images": (
                gallery_with_images
            ),
            "gallery_sampling": (
                args.sampling
            ),
            "gallery_sampling_seed": (
                args.seed
            ),
            "gallery_limit": (
                args.max_gallery_items
            ),
            "mean_confidence": round(
                mean_confidence,
                6,
            ),
            "mean_bbox_area": round(
                mean_bbox_area,
                8,
            ),
            "crop_thumbnail_size": list(
                CROP_THUMBNAIL_SIZE
            ),
            "crop_webp_quality": (
                CROP_WEBP_QUALITY
            ),
            "page_thumbnail_size": list(
                PAGE_THUMBNAIL_SIZE
            ),
            "page_webp_quality": (
                PAGE_WEBP_QUALITY
            ),
        },
        "pages": compact_pages,
        "spatial_detections": (
            spatial_detections
        ),
        "gallery": gallery,
        "metadata": {
            "quality": (
                metadata_quality
            ),
            "records_by_year": [
                {
                    "year": year,
                    "records": count,
                }
                for year, count
                in sorted(
                    metadata_years.items(),
                    key=lambda item: (
                        item[0] == "Unknown",
                        item[0],
                    ),
                )
            ],
            "pages_by_year": [
                {
                    "year": year,
                    "pages": count,
                }
                for year, count
                in sorted(
                    page_years.items(),
                    key=lambda item: (
                        item[0] == "Unknown",
                        item[0],
                    ),
                )
            ],
        },
        "classes": [
            {
                "class": class_name,
                "detections": count,
            }
            for class_name, count
            in detection_classes.most_common()
        ],
        "training": {
            "history": (
                training_results
            ),
            "manifest": (
                training_manifest
            ),
            "diagnostics": (
                copied_diagnostics
            ),
        },
        "inference": {
            "manifest": (
                inference_manifest
            ),
        },
        "preparation": {
            "detections_source": str(
                detections_path
            ),
            "pages_source": str(
                pages_path
            ),
            "metadata_source": str(
                metadata_path
            ),
            "training_directory": str(
                training_dir
            ),
            "validation_directory": (
                str(validation_dir)
                if validation_dir
                else None
            ),
            "sampling_strategy": (
                args.sampling
            ),
            "sampling_seed": (
                args.seed
            ),
            "gallery_limit": (
                args.max_gallery_items
            ),
        },
    }

    salvar_json(
        output_path,
        bundle,
    )

    # ========================================================
    # RESUMO FINAL
    # ========================================================

    print("\n" + "=" * 70)
    print("DASHBOARD DATA PREPARED")
    print("=" * 70)
    print(
        "Metadata records:            "
        f"{len(metadata)}"
    )
    print(
        "Processed pages:             "
        f"{len(compact_pages)}"
    )
    print(
        "Pages with detections:       "
        f"{pages_with_detections}"
    )
    print(
        "Spatial detections:          "
        f"{len(spatial_detections)}"
    )
    print(
        "Available detections:        "
        f"{len(detections)}"
    )
    print(
        "Gallery items:               "
        f"{len(gallery)}"
    )
    print(
        "Gallery items with images:   "
        f"{gallery_with_images}"
    )
    print(
        "Gallery sampling:            "
        f"{args.sampling}"
    )
    print(
        "Sampling seed:               "
        f"{args.seed}"
    )
    print(
        "Training epochs loaded:      "
        f"{len(training_results)}"
    )
    print(
        "Diagnostics copied:          "
        f"{len(copied_diagnostics)}"
    )
    print()
    print(
        f"Output JSON: {output_path}"
    )
    print(
        f"Assets: {assets}"
    )
    print("=" * 70)

    if gallery_with_images < len(gallery):
        missing = (
            len(gallery)
            - gallery_with_images
        )

        print(
            "AVISO: "
            f"{missing} itens da galeria não tiveram "
            "thumbnail gerado. Verifique os caminhos "
            "dos recortes originais."
        )


if __name__ == "__main__":
    main()
