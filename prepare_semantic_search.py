"""
Prepara um índice estático SigLIP2 para busca semântica.

A busca será executada inteiramente no navegador e permitirá:

    - consulta textual -> imagens semelhantes;
    - imagem enviada -> imagens semelhantes.

Modelo Python:
    google/siglip2-base-patch16-224

Modelo ONNX no navegador:
    onnx-community/siglip2-base-patch16-224-ONNX

Entrada:
    dashboard_data.json

Saídas:
    semantic_index_siglip2.f32
    semantic_index_siglip2_metadata.json

Uso:
    python prepare_semantic_search.py \
        --dashboard-data dashboard_data.json \
        --output-index semantic_index_siglip2.f32 \
        --output-metadata semantic_index_siglip2_metadata.json \
        --batch-size 8
"""

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor


# ============================================================
# CONFIGURAÇÃO
# ============================================================

PYTHON_MODEL_ID = (
    "openai/clip-vit-base-patch32"
)

BROWSER_MODEL_ID = (
    "Xenova/clip-vit-base-patch32"
)

DEFAULT_BATCH_SIZE = 8


# ============================================================
# ARGUMENTOS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Gera um índice SigLIP2 estático "
            "para busca semântica no GitHub Pages."
        )
    )

    parser.add_argument(
        "--dashboard-data",
        type=Path,
        default=Path(
            "dashboard_data.json"
        ),
    )

    parser.add_argument(
        "--output-index",
        type=Path,
        default=Path(
            "semantic_index_siglip2.f32"
        ),
    )

    parser.add_argument(
        "--output-metadata",
        type=Path,
        default=Path(
            "semantic_index_siglip2_metadata.json"
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    parser.add_argument(
        "--max-items",
        type=int,
        default=0,
        help=(
            "Quantidade máxima de imagens. "
            "Use 0 para toda a galeria pública."
        ),
    )

    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help=(
            "Confiança YOLO mínima entre 0 e 1."
        ),
    )

    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error(
            "--batch-size deve ser pelo menos 1"
        )

    if args.max_items < 0:
        parser.error(
            "--max-items deve ser maior ou igual a zero"
        )

    if not 0 <= args.min_confidence <= 1:
        parser.error(
            "--min-confidence deve estar entre 0 e 1"
        )

    return args


# ============================================================
# UTILITÁRIOS
# ============================================================

def detectar_device():
    if torch.cuda.is_available():
        print("Usando GPU NVIDIA via CUDA")
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        print("Usando Apple Silicon via MPS")
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


def carregar_imagem(caminho):
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


def limpar_memoria_device(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()

    elif device.type == "mps":
        try:
            torch.mps.empty_cache()
        except AttributeError:
            pass


def converter_saida_para_tensor(
    saida,
):
    """
    Converte diferentes formatos retornados pelo Transformers
    para um tensor de embeddings.
    """
    if isinstance(
        saida,
        torch.Tensor,
    ):
        return saida

    for atributo in (
        "image_embeds",
        "pooler_output",
        "text_embeds",
    ):
        valor = getattr(
            saida,
            atributo,
            None,
        )

        if isinstance(
            valor,
            torch.Tensor,
        ):
            return valor

    if isinstance(
        saida,
        (list, tuple),
    ):
        for valor in saida:
            if isinstance(
                valor,
                torch.Tensor,
            ):
                return valor

    raise RuntimeError(
        "Não foi possível localizar os embeddings "
        "na saída do modelo."
    )


# ============================================================
# REGISTROS
# ============================================================

def preparar_registros(
    dashboard,
    dashboard_directory,
    max_items,
    min_confidence,
):
    gallery = dashboard.get(
        "gallery",
        [],
    )

    if not isinstance(
        gallery,
        list,
    ):
        raise ValueError(
            "dashboard_data.json não contém "
            "uma galeria válida."
        )

    registros = []

    ignorados_confianca = 0
    ignorados_imagem = 0

    for item in gallery:
        if not isinstance(
            item,
            dict,
        ):
            continue

        confidence = numero_finito(
            item.get("confidence"),
            0,
        )

        if confidence < min_confidence:
            ignorados_confianca += 1
            continue

        crop_image = item.get(
            "crop_image"
        )

        if not crop_image:
            ignorados_imagem += 1
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

        if (
            not image_path.exists()
            or not image_path.is_file()
        ):
            print(
                "Warning: imagem não encontrada: "
                f"{image_path}"
            )

            ignorados_imagem += 1
            continue

        registro = dict(item)

        registro[
            "_image_path"
        ] = str(image_path)

        registros.append(registro)

        if (
            max_items > 0
            and len(registros) >= max_items
        ):
            break

    print(
        f"Itens válidos: {len(registros)}"
    )

    print(
        "Ignorados por confiança: "
        f"{ignorados_confianca}"
    )

    print(
        "Ignorados por imagem ausente: "
        f"{ignorados_imagem}"
    )

    return registros


# ============================================================
# EMBEDDINGS SIGLIP2
# ============================================================

def extrair_embeddings(
    registros,
    model,
    processor,
    device,
    batch_size,
):
    embeddings = []
    registros_processados = []

    for inicio in tqdm(
        range(
            0,
            len(registros),
            batch_size,
        ),
        desc="Gerando embeddings SigLIP2",
    ):
        lote = registros[
            inicio:
            inicio + batch_size
        ]

        imagens = []
        lote_valido = []

        for registro in lote:
            image_path = Path(
                registro[
                    "_image_path"
                ]
            )

            try:
                imagem = carregar_imagem(
                    image_path
                )

                imagens.append(imagem)
                lote_valido.append(registro)

            except Exception as erro:
                print(
                    "Warning: falha ao abrir "
                    f"{image_path}: {erro}"
                )

        if not imagens:
            continue

        inputs = processor(
            images=imagens,
            return_tensors="pt",
        )

        inputs = {
            chave: valor.to(device)
            for chave, valor
            in inputs.items()
            if isinstance(
                valor,
                torch.Tensor,
            )
        }

        with torch.inference_mode():
            saida = model.get_image_features(
                **inputs
            )

            features = (
                converter_saida_para_tensor(
                    saida
                )
            )

            features = features.float()

            features = (
                features
                / features.norm(
                    dim=-1,
                    keepdim=True,
                ).clamp(
                    min=1e-12
                )
            )

            batch_embeddings = (
                features
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

        del inputs
        del features
        del batch_embeddings

        limpar_memoria_device(device)

    if not embeddings:
        raise RuntimeError(
            "Nenhum embedding SigLIP2 foi gerado."
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
            "Embeddings e registros possuem "
            "tamanhos diferentes."
        )

    return (
        embeddings,
        registros_processados,
    )


# ============================================================
# METADADOS PÚBLICOS
# ============================================================

def criar_metadados_publicos(
    registros,
    embedding_dimension,
    min_confidence,
):
    items = []

    for index, registro in enumerate(
        registros
    ):
        items.append({
            "index": index,
            "id": registro.get("id"),
            "page_id": registro.get(
                "page_id"
            ),
            "year": registro.get("year"),
            "page": registro.get("page"),
            "issue": registro.get("issue"),
            "journal": registro.get(
                "journal"
            ),
            "confidence": numero_finito(
                registro.get(
                    "confidence"
                ),
                0,
            ),
            "crop_image": registro.get(
                "crop_image"
            ),
            "page_image": registro.get(
                "page_image"
            ),
            "url": registro.get("url"),
        })

    return {
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "model": {
            "family": "SigLIP2",
            "python_model": (
                PYTHON_MODEL_ID
            ),
            "browser_model": (
                BROWSER_MODEL_ID
            ),
            "dimension": int(
                embedding_dimension
            ),
            "normalization": "L2",
            "binary_dtype": (
                "float32-little-endian"
            ),
        },
        "selection": {
            "minimum_yolo_confidence": (
                min_confidence
            ),
            "items": len(items),
            "scope": (
                "Public dashboard gallery sample"
            ),
        },
        "items": items,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    dashboard_path = (
        args.dashboard_data
        .expanduser()
        .resolve()
    )

    output_index = (
        args.output_index
        .expanduser()
        .resolve()
    )

    output_metadata = (
        args.output_metadata
        .expanduser()
        .resolve()
    )

    dashboard = carregar_json(
        dashboard_path
    )

    registros = preparar_registros(
        dashboard=dashboard,
        dashboard_directory=(
            dashboard_path.parent
        ),
        max_items=args.max_items,
        min_confidence=(
            args.min_confidence
        ),
    )

    if not registros:
        raise ValueError(
            "Nenhuma imagem disponível "
            "para a busca semântica."
        )

    device = detectar_device()

    print("\n" + "=" * 70)
    print("SIGLIP2 SEMANTIC SEARCH INDEX")
    print("=" * 70)
    print(
        f"Modelo Python: {PYTHON_MODEL_ID}"
    )
    print(
        f"Modelo browser: {BROWSER_MODEL_ID}"
    )
    print(f"Device: {device}")
    print(f"Imagens: {len(registros)}")
    print("=" * 70)

    processor = (
        AutoProcessor
        .from_pretrained(
            PYTHON_MODEL_ID
        )
    )

    model = (
        AutoModel
        .from_pretrained(
            PYTHON_MODEL_ID
        )
    )

    model = model.to(device)
    model.eval()

    (
        embeddings,
        registros_processados,
    ) = extrair_embeddings(
        registros=registros,
        model=model,
        processor=processor,
        device=device,
        batch_size=args.batch_size,
    )

    output_index.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    embeddings_le = embeddings.astype(
        "<f4",
        copy=False,
    )

    embeddings_le.tofile(
        output_index
    )

    public_metadata = (
        criar_metadados_publicos(
            registros=(
                registros_processados
            ),
            embedding_dimension=(
                embeddings.shape[1]
            ),
            min_confidence=(
                args.min_confidence
            ),
        )
    )

    salvar_json(
        output_metadata,
        public_metadata,
    )

    expected_bytes = (
        embeddings.shape[0]
        * embeddings.shape[1]
        * 4
    )

    actual_bytes = (
        output_index.stat().st_size
    )

    if actual_bytes != expected_bytes:
        raise RuntimeError(
            "Tamanho inesperado no índice binário. "
            f"Esperado: {expected_bytes}; "
            f"encontrado: {actual_bytes}."
        )

    print("\n" + "=" * 70)
    print("SIGLIP2 INDEX COMPLETED")
    print("=" * 70)
    print(
        "Itens:                       "
        f"{embeddings.shape[0]}"
    )
    print(
        "Dimensão:                    "
        f"{embeddings.shape[1]}"
    )
    print(
        "Tamanho:                     "
        f"{actual_bytes / 1024 / 1024:.2f} MB"
    )
    print(f"Índice: {output_index}")
    print(f"Metadados: {output_metadata}")
    print("=" * 70)


if __name__ == "__main__":
    main()