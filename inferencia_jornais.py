"""
Inferência YOLO Detect — A Illustração.

Executa a inferência nas páginas do metadata_enriquecido.json,
limitando opcionalmente o processamento por ano.

Saídas:
    - um PNG para cada elemento detectado;
    - um preview JPG de baixa qualidade para todas as páginas processadas;
    - JSON e CSV com um registro para cada elemento extraído;
    - JSON com um registro para cada página processada;
    - JSON com parâmetros e resumo da execução;
    - JSON com erros encontrados.

Uso:
    python inferencia_detect.py --conf 0.50 --ano-max 1930

Exemplos:
    --conf 0.30       aceita detecções com pelo menos 30%
    --conf 0.50       aceita detecções com pelo menos 50%
    --conf 0.70       aceita detecções com pelo menos 70%
    --ano-max 1930    processa páginas até 1930, inclusive
    --ano-max 1925    processa páginas até 1925, inclusive

Observações:
    - Registros com ano maior que --ano-max são ignorados.
    - Registros sem ano válido também são ignorados.
    - Como CLEAN_OUTPUT = True, resultados anteriores são apagados.
"""

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import cv2
import ultralytics
from tqdm import tqdm
from ultralytics import YOLO


# ============================================================
# CONFIGURAÇÃO FIXA
# ============================================================

MODEL_PATH = Path(
    "/Users/luanamoraescosta/DH/ilustracao/finetune/"
    "runs/detect/runs_jornal/layout_jornal_det/weights/best.pt"
)

METADATA_PATH = Path(
    "/Users/luanamoraescosta/DH/ilustracao/"
    "ilustracaobrasil/metadata_enriquecido.json"
)

OUTPUT_PREFIX = Path(
    "/Users/luanamoraescosta/DH/ilustracao/"
    "ilustracaobrasil/resultados_inferencia"
)

IMAGE_KEY = "imagem_local"

METADATA_KEYS = [
    "bib",
    "num",
    "total_paginas_edicao",
    "pdf_local",
]

# Inferência
DEVICE = "mps"
IMGSZ = 1280
IOU_NMS = 0.50

# Recortes
PADDING = 20

# Compressão máxima do PNG, sem perda de qualidade.
# Valores permitidos pelo OpenCV: 0 a 9.
PNG_COMPRESSION = 0

# Preview leve das páginas
PREVIEW_MAX_WIDTH = 800
PREVIEW_JPEG_QUALITY = 10

# Apaga os resultados da execução anterior.
CLEAN_OUTPUT = True

# Ano máximo padrão, inclusive.
DEFAULT_ANO_MAX = 1920


# ============================================================
# ARGUMENTOS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Inferência YOLO Detect em A Illustração, "
            "com filtro temporal."
        )
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.50,
        help="confiança mínima da detecção, entre 0 e 1",
    )

    parser.add_argument(
        "--ano-max",
        type=int,
        default=DEFAULT_ANO_MAX,
        help=(
            "ano máximo incluído na inferência. "
            "O valor é inclusivo; por exemplo, 1930 inclui 1930"
        ),
    )

    args = parser.parse_args()

    if not 0 <= args.conf <= 1:
        parser.error("--conf deve estar entre 0 e 1")

    if args.ano_max < 1:
        parser.error("--ano-max deve ser um ano válido")

    return args


# ============================================================
# CONVERSÃO DO ANO
# ============================================================

def converter_ano(valor):
    """
    Converte um valor de ano para inteiro.

    Aceita:
        1930
        "1930"
        1930.0
        "1930.0"
        "Ano 1930"

    Retorna None quando não for possível encontrar um ano válido.
    """
    if valor is None:
        return None

    try:
        ano = int(float(valor))

        if 1000 <= ano <= 2999:
            return ano

    except (TypeError, ValueError):
        pass

    correspondencia = re.search(
        r"\b(1[0-9]{3}|2[0-9]{3})\b",
        str(valor),
    )

    if correspondencia:
        return int(
            correspondencia.group(1)
        )

    return None


# ============================================================
# NOMES ÚNICOS
# ============================================================

def texto_seguro(valor, limite=60):
    """
    Transforma um valor em texto seguro para nome de arquivo.
    """
    valor = str(valor or "unknown")

    valor = unicodedata.normalize(
        "NFKD",
        valor,
    )

    valor = (
        valor
        .encode("ascii", "ignore")
        .decode("ascii")
    )

    valor = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        valor,
    )

    valor = valor.strip("._-")

    if not valor:
        valor = "unknown"

    return valor[:limite]


def criar_id_pagina(item, image_path):
    """
    Cria um identificador único para cada página.

    O hash do caminho evita colisões entre páginas com o mesmo
    ano, número de página ou nome de arquivo.
    """
    revista = (
        item.get("nome")
        or item.get("jornal")
        or "jornal"
    )

    edicao = (
        item.get("edicao")
        or item.get("num")
        or Path(
            item.get("pdf_local", "")
        ).stem
        or image_path.parent.name
        or "edicao"
    )

    ano = converter_ano(
        item.get("ano")
    )

    if ano is None:
        ano = "unknown"

    pagina = item.get(
        "pagina",
        "unknown",
    )

    hash_caminho = hashlib.sha1(
        str(
            image_path.resolve()
        ).encode("utf-8")
    ).hexdigest()[:10]

    return "_".join([
        texto_seguro(
            revista,
            30,
        ),
        texto_seguro(
            ano,
            10,
        ),
        texto_seguro(
            edicao,
            40,
        ),
        (
            "pagina_"
            f"{texto_seguro(pagina, 10)}"
        ),
        texto_seguro(
            image_path.stem,
            40,
        ),
        hash_caminho,
    ])


# ============================================================
# CAMINHOS DE SAÍDA
# ============================================================

def obter_saidas():
    prefixo = str(OUTPUT_PREFIX)

    return {
        "json": Path(
            f"{prefixo}_resultados.json"
        ),
        "csv": Path(
            f"{prefixo}_resultados.csv"
        ),
        "paginas": Path(
            f"{prefixo}_paginas.json"
        ),
        "execucao": Path(
            f"{prefixo}_execucao.json"
        ),
        "erros": Path(
            f"{prefixo}_erros.json"
        ),
        "previews": Path(
            f"{prefixo}_imagens_detectadas"
        ),
        "elementos": Path(
            f"{prefixo}_elementos_extraidos"
        ),
    }


def preparar_saidas():
    saidas = obter_saidas()

    OUTPUT_PREFIX.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if CLEAN_OUTPUT:
        print("\nLimpando resultados anteriores...")

        for chave in (
            "json",
            "csv",
            "paginas",
            "execucao",
            "erros",
        ):
            arquivo = saidas[chave]

            if arquivo.exists():
                arquivo.unlink()
                print(f"  Removido: {arquivo}")

        for chave in (
            "previews",
            "elementos",
        ):
            pasta = saidas[chave]

            if pasta.exists():
                shutil.rmtree(pasta)
                print(f"  Removida: {pasta}")

    saidas["previews"].mkdir(
        parents=True,
        exist_ok=True,
    )

    saidas["elementos"].mkdir(
        parents=True,
        exist_ok=True,
    )

    return saidas


# ============================================================
# CLASSES DO MODELO
# ============================================================

def obter_nome_classe(model, classe):
    nomes = model.names

    if isinstance(nomes, dict):
        return nomes.get(
            classe,
            str(classe),
        )

    if isinstance(
        nomes,
        (list, tuple),
    ):
        if 0 <= classe < len(nomes):
            return nomes[classe]

    return str(classe)


# ============================================================
# PREVIEW DAS PÁGINAS
# ============================================================

def salvar_preview(
    img,
    boxes,
    model,
    output_dir,
    page_id,
):
    """
    Salva um preview JPG leve.

    A função é chamada para todas as páginas processadas,
    incluindo aquelas sem detecção.
    """
    altura_original, largura_original = img.shape[:2]

    escala = min(
        1.0,
        PREVIEW_MAX_WIDTH / largura_original,
    )

    if escala < 1:
        nova_largura = PREVIEW_MAX_WIDTH

        nova_altura = max(
            1,
            round(
                altura_original * escala
            ),
        )

        preview = cv2.resize(
            img,
            (
                nova_largura,
                nova_altura,
            ),
            interpolation=cv2.INTER_AREA,
        )

    else:
        preview = img.copy()

    quantidade = (
        len(boxes)
        if boxes is not None
        else 0
    )

    espessura = 2
    tamanho_fonte = 0.45

    if quantidade == 0:
        texto = "SEM DETECCOES"

        cv2.rectangle(
            preview,
            (5, 5),
            (185, 32),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            preview,
            texto,
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    else:
        for indice, box in enumerate(boxes):
            x1, y1, x2, y2 = (
                box.xyxy[0]
                .detach()
                .cpu()
                .tolist()
            )

            x1 = round(x1 * escala)
            y1 = round(y1 * escala)
            x2 = round(x2 * escala)
            y2 = round(y2 * escala)

            confianca = float(
                box.conf[0]
                .detach()
                .cpu()
            )

            classe = int(
                box.cls[0]
                .detach()
                .cpu()
            )

            nome_classe = obter_nome_classe(
                model,
                classe,
            )

            cv2.rectangle(
                preview,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                espessura,
            )

            label = (
                f"{indice:04d} | "
                f"{nome_classe} | "
                f"{confianca * 100:.1f}%"
            )

            (
                largura_texto,
                altura_texto,
            ), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                tamanho_fonte,
                espessura,
            )

            texto_y = max(
                altura_texto + 6,
                y1 - 5,
            )

            cv2.rectangle(
                preview,
                (
                    x1,
                    texto_y
                    - altura_texto
                    - 5,
                ),
                (
                    x1
                    + largura_texto
                    + 5,
                    texto_y
                    + baseline,
                ),
                (0, 0, 0),
                -1,
            )

            cv2.putText(
                preview,
                label,
                (
                    x1 + 2,
                    texto_y,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                tamanho_fonte,
                (0, 255, 0),
                espessura,
                cv2.LINE_AA,
            )

    output_path = (
        output_dir
        / f"{page_id}_detectada.jpg"
    )

    sucesso = cv2.imwrite(
        str(output_path),
        preview,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            PREVIEW_JPEG_QUALITY,
        ],
    )

    if not sucesso:
        raise RuntimeError(
            "Não foi possível salvar o preview: "
            f"{output_path}"
        )

    return str(output_path)


# ============================================================
# EXTRAÇÃO DOS ELEMENTOS
# ============================================================

def extrair_elementos(
    img,
    boxes,
    output_dir,
    page_id,
):
    """
    Recorta cada bounding box com padding.

    Os elementos são salvos:
        - em PNG;
        - na resolução original;
        - sem redimensionamento;
        - com compressão máxima sem perda de qualidade.
    """
    altura, largura = img.shape[:2]
    elementos = []

    if boxes is None or len(boxes) == 0:
        return elementos

    for indice, box in enumerate(boxes):
        x1, y1, x2, y2 = (
            box.xyxy[0]
            .detach()
            .cpu()
            .tolist()
        )

        bbox_original = [
            float(x1),
            float(y1),
            float(x2),
            float(y2),
        ]

        # Floor no início e ceil no final para evitar
        # cortar partes da detecção.
        bbox_x1 = max(
            0,
            min(
                largura,
                math.floor(x1),
            ),
        )

        bbox_y1 = max(
            0,
            min(
                altura,
                math.floor(y1),
            ),
        )

        bbox_x2 = max(
            0,
            min(
                largura,
                math.ceil(x2),
            ),
        )

        bbox_y2 = max(
            0,
            min(
                altura,
                math.ceil(y2),
            ),
        )

        # Adiciona padding ao recorte.
        crop_x1 = max(
            0,
            bbox_x1 - PADDING,
        )

        crop_y1 = max(
            0,
            bbox_y1 - PADDING,
        )

        crop_x2 = min(
            largura,
            bbox_x2 + PADDING,
        )

        crop_y2 = min(
            altura,
            bbox_y2 + PADDING,
        )

        bbox_recorte = [
            crop_x1,
            crop_y1,
            crop_x2,
            crop_y2,
        ]

        info = {
            "caminho": None,
            "bbox": bbox_original,
            "bbox_recorte": bbox_recorte,
            "largura_recorte": (
                crop_x2 - crop_x1
            ),
            "altura_recorte": (
                crop_y2 - crop_y1
            ),
            "erro": None,
        }

        if (
            crop_x2 <= crop_x1
            or crop_y2 <= crop_y1
        ):
            info["erro"] = (
                "coordenadas de recorte inválidas"
            )

            elementos.append(info)
            continue

        cropped = img[
            crop_y1:crop_y2,
            crop_x1:crop_x2,
        ]

        if cropped.size == 0:
            info["erro"] = "recorte vazio"
            elementos.append(info)
            continue

        output_path = (
            output_dir
            / (
                f"{page_id}_"
                f"elemento_{indice:04d}.png"
            )
        )

        sucesso = cv2.imwrite(
            str(output_path),
            cropped,
            [
                cv2.IMWRITE_PNG_COMPRESSION,
                PNG_COMPRESSION,
            ],
        )

        if sucesso:
            info["caminho"] = str(
                output_path
            )
        else:
            info["erro"] = (
                "cv2.imwrite retornou False"
            )

        elementos.append(info)

    return elementos


# ============================================================
# JSON E CSV
# ============================================================

def salvar_json(caminho, dados):
    caminho.write_text(
        json.dumps(
            dados,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def salvar_csv(caminho, resultados):
    campos_base = [
        "id",
        "page_id",
        "indice_deteccao",
        "revista",
        "ano",
        "pagina",
        "url",
        "imagem",
        "imagem_detectada",
        "elemento_extraido",
        "confianca",
        "confianca_percentual",
        "classe",
        "nome_classe",
        "bbox",
        "bbox_recorte",
        "padding",
        "largura_pagina",
        "altura_pagina",
        "area_pagina",
        "largura_bbox",
        "altura_bbox",
        "area_bbox",
        "area_bbox_percentual_pagina",
        "largura_recorte",
        "altura_recorte",
        "area_recorte",
        "area_recorte_percentual_pagina",
    ]

    linhas = []

    for registro in resultados:
        linha = registro.copy()

        for chave, valor in linha.items():
            if isinstance(
                valor,
                (list, dict),
            ):
                linha[chave] = json.dumps(
                    valor,
                    ensure_ascii=False,
                )

        linhas.append(linha)

    fieldnames = list(campos_base)

    for linha in linhas:
        for chave in linha:
            if chave not in fieldnames:
                fieldnames.append(chave)

    with caminho.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as arquivo:
        writer = csv.DictWriter(
            arquivo,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        if linhas:
            writer.writerows(linhas)


# ============================================================
# FILTRO DOS METADADOS
# ============================================================

def filtrar_metadata_por_ano(
    metadata,
    ano_maximo,
):
    """
    Filtra os metadados pelo ano máximo.

    Retorna:
        - registros selecionados com seus índices originais;
        - quantidade de registros acima do ano máximo;
        - quantidade de registros sem ano válido;
        - quantidade de registros inválidos.
    """
    metadata_selecionado = []

    paginas_apos_ano_maximo = 0
    paginas_sem_ano_valido = 0
    registros_invalidos = 0

    for indice_original, item in enumerate(metadata):
        if not isinstance(item, dict):
            registros_invalidos += 1
            continue

        ano_item = converter_ano(
            item.get("ano")
        )

        if ano_item is None:
            paginas_sem_ano_valido += 1
            continue

        if ano_item > ano_maximo:
            paginas_apos_ano_maximo += 1
            continue

        metadata_selecionado.append(
            (
                indice_original,
                item,
                ano_item,
            )
        )

    return {
        "selecionados": metadata_selecionado,
        "apos_ano_maximo": paginas_apos_ano_maximo,
        "sem_ano_valido": paginas_sem_ano_valido,
        "registros_invalidos": registros_invalidos,
    }


# ============================================================
# PROCESSAMENTO
# ============================================================

def processar_dataset(
    model,
    confianca_minima,
    ano_maximo,
):
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            "Metadados não encontrados: "
            f"{METADATA_PATH}"
        )

    with METADATA_PATH.open(
        "r",
        encoding="utf-8",
    ) as arquivo:
        metadata = json.load(arquivo)

    if not isinstance(metadata, list):
        raise ValueError(
            "O arquivo de metadados precisa conter uma lista."
        )

    total_metadata_original = len(metadata)

    filtro = filtrar_metadata_por_ano(
        metadata=metadata,
        ano_maximo=ano_maximo,
    )

    metadata_selecionado = filtro[
        "selecionados"
    ]

    paginas_apos_ano_maximo = filtro[
        "apos_ano_maximo"
    ]

    paginas_sem_ano_valido = filtro[
        "sem_ano_valido"
    ]

    registros_invalidos = filtro[
        "registros_invalidos"
    ]

    total_paginas_selecionadas = len(
        metadata_selecionado
    )

    if total_paginas_selecionadas == 0:
        raise ValueError(
            "Nenhuma página foi encontrada para o filtro "
            f"ano <= {ano_maximo}."
        )

    saidas = preparar_saidas()

    inicio_execucao = datetime.now(
        timezone.utc
    )

    print("\n" + "=" * 70)
    print("CONFIGURAÇÃO DA INFERÊNCIA")
    print("=" * 70)
    print(f"Modelo: {MODEL_PATH}")
    print(f"Metadados: {METADATA_PATH}")
    print(
        "Páginas nos metadados: "
        f"{total_metadata_original}"
    )
    print(
        f"Filtro temporal: ano <= {ano_maximo}"
    )
    print(
        "Páginas selecionadas: "
        f"{total_paginas_selecionadas}"
    )
    print(
        f"Páginas após {ano_maximo}: "
        f"{paginas_apos_ano_maximo}"
    )
    print(
        "Páginas sem ano válido: "
        f"{paginas_sem_ano_valido}"
    )
    print(
        "Registros inválidos: "
        f"{registros_invalidos}"
    )
    print(
        "Confiança mínima: "
        f"{confianca_minima:.2f}"
    )
    print(f"Dispositivo: {DEVICE}")
    print(f"Resolução YOLO: {IMGSZ}")
    print(f"IOU NMS: {IOU_NMS:.2f}")
    print(
        f"Padding dos recortes: {PADDING}px"
    )
    print(
        "PNG: resolução original, "
        f"compressão {PNG_COMPRESSION}"
    )
    print(
        "Preview: JPG, largura máxima "
        f"{PREVIEW_MAX_WIDTH}px, "
        f"qualidade {PREVIEW_JPEG_QUALITY}"
    )
    print("=" * 70)

    resultados = []
    paginas = []
    erros = []

    paginas_processadas = 0
    paginas_com_deteccao = 0
    paginas_sem_deteccao = 0
    previews_salvos = 0
    recortes_salvos = 0

    barra = tqdm(
        metadata_selecionado,
        total=total_paginas_selecionadas,
        desc=f"Processando até {ano_maximo}",
    )

    for (
        indice_metadata,
        item,
        ano_item,
    ) in barra:
        image_path_value = item.get(
            IMAGE_KEY
        )

        if not image_path_value:
            erros.append({
                "indice_metadata": indice_metadata,
                "ano": ano_item,
                "erro": (
                    f"campo ausente: {IMAGE_KEY}"
                ),
            })

            continue

        image_path = Path(
            image_path_value
        ).expanduser()

        if not image_path.exists():
            erros.append({
                "indice_metadata": indice_metadata,
                "ano": ano_item,
                "imagem": str(image_path),
                "erro": "imagem não encontrada",
            })

            continue

        try:
            img = cv2.imread(
                str(image_path)
            )

            if img is None:
                raise RuntimeError(
                    "OpenCV não conseguiu abrir a imagem"
                )

            (
                altura_pagina,
                largura_pagina,
            ) = img.shape[:2]

            area_pagina = (
                largura_pagina
                * altura_pagina
            )

            page_id = criar_id_pagina(
                item,
                image_path,
            )

            # =================================================
            # INFERÊNCIA YOLO
            # =================================================

            result = model.predict(
                source=img,
                conf=confianca_minima,
                iou=IOU_NMS,
                imgsz=IMGSZ,
                device=DEVICE,
                verbose=False,
            )[0]

            boxes = result.boxes

            quantidade_deteccoes = (
                len(boxes)
                if boxes is not None
                else 0
            )

            paginas_processadas += 1

            # =================================================
            # PREVIEW DA PÁGINA
            # =================================================

            preview_path = salvar_preview(
                img=img,
                boxes=boxes,
                model=model,
                output_dir=saidas["previews"],
                page_id=page_id,
            )

            previews_salvos += 1

            # =================================================
            # REGISTRO DA PÁGINA
            # =================================================

            registro_pagina = {
                "page_id": page_id,
                "indice_metadata": indice_metadata,
                "revista": (
                    item.get("nome")
                    or item.get("jornal")
                    or "unknown"
                ),
                "ano": ano_item,
                "pagina": item.get(
                    "pagina",
                    "unknown",
                ),
                "url": item.get(
                    "url_origem",
                    "",
                ),
                "imagem": str(image_path),
                "imagem_detectada": preview_path,
                "largura_pagina": largura_pagina,
                "altura_pagina": altura_pagina,
                "area_pagina": area_pagina,
                "quantidade_deteccoes": (
                    quantidade_deteccoes
                ),
                "tem_deteccao": (
                    quantidade_deteccoes > 0
                ),
                "status": "processed",
            }

            for chave in METADATA_KEYS:
                if chave in item:
                    registro_pagina[chave] = (
                        item[chave]
                    )

            paginas.append(
                registro_pagina
            )

            if quantidade_deteccoes == 0:
                paginas_sem_deteccao += 1

                barra.set_postfix({
                    "páginas": paginas_processadas,
                    "previews": previews_salvos,
                    "elementos": recortes_salvos,
                })

                continue

            paginas_com_deteccao += 1

            # =================================================
            # EXTRAÇÃO DOS ELEMENTOS
            # =================================================

            elementos = extrair_elementos(
                img=img,
                boxes=boxes,
                output_dir=saidas["elementos"],
                page_id=page_id,
            )

            for (
                indice_deteccao,
                box,
            ) in enumerate(boxes):
                elemento = elementos[
                    indice_deteccao
                ]

                if elemento["caminho"] is None:
                    erros.append({
                        "indice_metadata": (
                            indice_metadata
                        ),
                        "page_id": page_id,
                        "ano": ano_item,
                        "imagem": str(image_path),
                        "deteccao": (
                            indice_deteccao
                        ),
                        "erro": elemento["erro"],
                    })

                    continue

                confianca = float(
                    box.conf[0]
                    .detach()
                    .cpu()
                )

                classe = int(
                    box.cls[0]
                    .detach()
                    .cpu()
                )

                nome_classe = obter_nome_classe(
                    model,
                    classe,
                )

                detection_id = (
                    f"{page_id}_"
                    f"det_{indice_deteccao:04d}"
                )

                bbox = elemento["bbox"]

                largura_bbox = max(
                    0.0,
                    bbox[2] - bbox[0],
                )

                altura_bbox = max(
                    0.0,
                    bbox[3] - bbox[1],
                )

                area_bbox = (
                    largura_bbox
                    * altura_bbox
                )

                area_recorte = (
                    elemento["largura_recorte"]
                    * elemento["altura_recorte"]
                )

                if area_pagina > 0:
                    area_bbox_percentual = (
                        area_bbox
                        / area_pagina
                        * 100
                    )

                    area_recorte_percentual = (
                        area_recorte
                        / area_pagina
                        * 100
                    )

                else:
                    area_bbox_percentual = 0
                    area_recorte_percentual = 0

                registro = {
                    "id": detection_id,
                    "page_id": page_id,
                    "indice_deteccao": (
                        indice_deteccao
                    ),
                    "revista": (
                        item.get("nome")
                        or item.get("jornal")
                        or "unknown"
                    ),
                    "ano": ano_item,
                    "pagina": item.get(
                        "pagina",
                        "unknown",
                    ),
                    "url": item.get(
                        "url_origem",
                        "",
                    ),
                    "imagem": str(image_path),
                    "imagem_detectada": (
                        preview_path
                    ),
                    "elemento_extraido": (
                        elemento["caminho"]
                    ),
                    "confianca": confianca,
                    "confianca_percentual": round(
                        confianca * 100,
                        4,
                    ),
                    "classe": classe,
                    "nome_classe": nome_classe,
                    "bbox": elemento["bbox"],
                    "bbox_recorte": elemento[
                        "bbox_recorte"
                    ],
                    "padding": PADDING,
                    "largura_pagina": (
                        largura_pagina
                    ),
                    "altura_pagina": (
                        altura_pagina
                    ),
                    "area_pagina": area_pagina,
                    "largura_bbox": largura_bbox,
                    "altura_bbox": altura_bbox,
                    "area_bbox": area_bbox,
                    "area_bbox_percentual_pagina": (
                        round(
                            area_bbox_percentual,
                            6,
                        )
                    ),
                    "largura_recorte": elemento[
                        "largura_recorte"
                    ],
                    "altura_recorte": elemento[
                        "altura_recorte"
                    ],
                    "area_recorte": area_recorte,
                    "area_recorte_percentual_pagina": (
                        round(
                            area_recorte_percentual,
                            6,
                        )
                    ),
                }

                for chave in METADATA_KEYS:
                    if chave in item:
                        registro[chave] = (
                            item[chave]
                        )

                resultados.append(
                    registro
                )

                recortes_salvos += 1

            barra.set_postfix({
                "páginas": paginas_processadas,
                "previews": previews_salvos,
                "elementos": recortes_salvos,
            })

        except Exception as erro:
            erros.append({
                "indice_metadata": indice_metadata,
                "ano": ano_item,
                "imagem": str(image_path),
                "erro": str(erro),
            })

    # ========================================================
    # SALVAR RESULTADOS
    # ========================================================

    fim_execucao = datetime.now(
        timezone.utc
    )

    duracao_segundos = (
        fim_execucao
        - inicio_execucao
    ).total_seconds()

    salvar_json(
        saidas["json"],
        resultados,
    )

    salvar_csv(
        saidas["csv"],
        resultados,
    )

    salvar_json(
        saidas["paginas"],
        paginas,
    )

    salvar_json(
        saidas["erros"],
        erros,
    )

    manifesto_execucao = {
        "created_at": (
            inicio_execucao.isoformat()
        ),
        "finished_at": (
            fim_execucao.isoformat()
        ),
        "duration_seconds": round(
            duracao_segundos,
            4,
        ),
        "model_path": str(
            MODEL_PATH.resolve()
        ),
        "metadata_path": str(
            METADATA_PATH.resolve()
        ),
        "output_prefix": str(
            OUTPUT_PREFIX.resolve()
        ),
        "device": DEVICE,
        "imgsz": IMGSZ,
        "confidence_threshold": (
            confianca_minima
        ),
        "iou_nms": IOU_NMS,
        "padding": PADDING,
        "png_compression": (
            PNG_COMPRESSION
        ),
        "preview_max_width": (
            PREVIEW_MAX_WIDTH
        ),
        "preview_jpeg_quality": (
            PREVIEW_JPEG_QUALITY
        ),
        "ultralytics_version": (
            ultralytics.__version__
        ),
        "pages_in_metadata": (
            total_metadata_original
        ),
        "year_filter": {
            "maximum_year_inclusive": (
                ano_maximo
            ),
            "pages_selected": (
                total_paginas_selecionadas
            ),
            "pages_after_maximum_year": (
                paginas_apos_ano_maximo
            ),
            "pages_without_valid_year": (
                paginas_sem_ano_valido
            ),
            "invalid_metadata_records": (
                registros_invalidos
            ),
        },
        "pages_processed": (
            paginas_processadas
        ),
        "pages_with_detections": (
            paginas_com_deteccao
        ),
        "pages_without_detections": (
            paginas_sem_deteccao
        ),
        "previews_saved": (
            previews_salvos
        ),
        "detections_saved": (
            recortes_salvos
        ),
        "errors": len(erros),
        "output_files": {
            "detections_json": str(
                saidas["json"].resolve()
            ),
            "detections_csv": str(
                saidas["csv"].resolve()
            ),
            "pages_json": str(
                saidas["paginas"].resolve()
            ),
            "errors_json": str(
                saidas["erros"].resolve()
            ),
            "previews_directory": str(
                saidas["previews"].resolve()
            ),
            "elements_directory": str(
                saidas["elementos"].resolve()
            ),
        },
    }

    salvar_json(
        saidas["execucao"],
        manifesto_execucao,
    )

    # ========================================================
    # RESUMO
    # ========================================================

    print("\n" + "=" * 70)
    print("RESUMO FINAL")
    print("=" * 70)
    print(
        "Páginas nos metadados:       "
        f"{total_metadata_original}"
    )
    print(
        "Ano máximo incluído:          "
        f"{ano_maximo}"
    )
    print(
        "Páginas selecionadas:         "
        f"{total_paginas_selecionadas}"
    )
    print(
        "Páginas após o ano máximo:    "
        f"{paginas_apos_ano_maximo}"
    )
    print(
        "Páginas sem ano válido:       "
        f"{paginas_sem_ano_valido}"
    )
    print(
        "Registros inválidos:          "
        f"{registros_invalidos}"
    )
    print(
        "Páginas processadas:          "
        f"{paginas_processadas}"
    )
    print(
        "Páginas com detecção:         "
        f"{paginas_com_deteccao}"
    )
    print(
        "Páginas sem detecção:         "
        f"{paginas_sem_deteccao}"
    )
    print(
        "Previews salvos:              "
        f"{previews_salvos}"
    )
    print(
        "Elementos PNG salvos:         "
        f"{recortes_salvos}"
    )
    print(
        "Erros:                        "
        f"{len(erros)}"
    )
    print(
        "Duração total:                "
        f"{duracao_segundos:.2f}s"
    )
    print()
    print(
        f"JSON de detecções: {saidas['json']}"
    )
    print(
        f"CSV de detecções: {saidas['csv']}"
    )
    print(
        f"JSON de páginas: {saidas['paginas']}"
    )
    print(
        "Manifesto da execução: "
        f"{saidas['execucao']}"
    )
    print(
        f"JSON de erros: {saidas['erros']}"
    )
    print(
        f"Previews: {saidas['previews']}"
    )
    print(
        f"Elementos: {saidas['elementos']}"
    )
    print("=" * 70)

    if previews_salvos != paginas_processadas:
        print(
            "AVISO: nem todas as páginas processadas "
            "tiveram o preview salvo. Consulte o arquivo "
            "de erros."
        )


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Modelo não encontrado: "
            f"{MODEL_PATH}"
        )

    print("=" * 70)
    print("CARREGANDO MODELO YOLO DETECT")
    print("=" * 70)
    print(MODEL_PATH)

    model = YOLO(
        str(MODEL_PATH)
    )

    processar_dataset(
        model=model,
        confianca_minima=args.conf,
        ano_maximo=args.ano_max,
    )


if __name__ == "__main__":
    main()