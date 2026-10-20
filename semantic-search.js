import {
    AutoProcessor,
    AutoTokenizer,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
    RawImage,
    env
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3";


/* ==========================================================
   CONFIGURATION
   ========================================================== */

env.allowLocalModels = false;
env.useBrowserCache = true;


const METADATA_URL =
    "semantic_index_metadata.json";

const INDEX_URL =
    "semantic_index.f32";

const MODEL_DTYPE = "q8";


/* ==========================================================
   APPLICATION STATE
   ========================================================== */

let searchMetadata = null;
let searchEmbeddings = null;

/*
 * Full per-query rankings are cached by the query vector
 * reference. The dot-product pass over the whole index is the
 * expensive step (13,944 images); filtering by year, confidence
 * and limit is cheap on top of a cached ranking.
 */
const rankingCache = new Map();

let tokenizer = null;
let processor = null;

let textModel = null;
let visionModel = null;

let textModelPromise = null;
let visionModelPromise = null;

let selectedImageFile = null;
let selectedImageURL = null;

let progressHideTimeout = null;


/* ==========================================================
   INITIALIZATION
   ========================================================== */

async function initializeSemanticSearch() {
    const status =
        document.getElementById(
            "semantic-status"
        );

    if (!status) {
        return;
    }

    try {
        status.textContent =
            "Loading the CLIP search index…";

        const [
            metadataResponse,
            indexResponse
        ] = await Promise.all([
            fetch(METADATA_URL),
            fetch(INDEX_URL)
        ]);

        if (!metadataResponse.ok) {
            throw new Error(
                "Could not load CLIP metadata: "
                + `HTTP ${metadataResponse.status}.`
            );
        }

        if (!indexResponse.ok) {
            throw new Error(
                "Could not load CLIP index: "
                + `HTTP ${indexResponse.status}.`
            );
        }

        searchMetadata =
            await metadataResponse.json();

        const indexBuffer =
            await indexResponse.arrayBuffer();

        searchEmbeddings =
            new Float32Array(
                indexBuffer
            );

        const dimension = Number(
            searchMetadata
                ?.model
                ?.dimension
        );

        const itemCount =
            searchMetadata
                ?.items
                ?.length
            || 0;

        if (
            !Number.isInteger(dimension)
            || dimension <= 0
        ) {
            throw new Error(
                "Invalid embedding dimension "
                + "in CLIP metadata."
            );
        }

        if (itemCount === 0) {
            throw new Error(
                "The CLIP index contains no items."
            );
        }

        const expectedValues =
            itemCount * dimension;

        if (
            searchEmbeddings.length
            !== expectedValues
        ) {
            throw new Error(
                "The CLIP index does not match "
                + "its metadata. "
                + `Expected ${expectedValues} values, `
                + `received ${searchEmbeddings.length}.`
            );
        }

        populateYears(
            searchMetadata.items
        );

        connectControls();

        status.textContent =
            `${itemCount.toLocaleString()} images indexed `
            + "with CLIP.";

    } catch (error) {
        console.error(
            "CLIP initialization failed:",
            error
        );

        status.textContent =
            "The CLIP search index "
            + "could not be loaded.";
    }
}

/* ==========================================================
   MODEL LOADING PROGRESS
   ========================================================== */

function getProgressElements() {
    return {
        container:
            document.getElementById(
                "semantic-model-progress"
            ),

        label:
            document.getElementById(
                "semantic-progress-label"
            ),

        percent:
            document.getElementById(
                "semantic-progress-percent"
            ),

        track:
            document.getElementById(
                "semantic-progress-track"
            ),

        fill:
            document.getElementById(
                "semantic-progress-fill"
            ),

        detail:
            document.getElementById(
                "semantic-progress-detail"
            )
    };
}


function formatDownloadSize(bytes) {
    const numeric = Number(bytes);

    if (
        !Number.isFinite(numeric)
        || numeric < 0
    ) {
        return null;
    }

    if (numeric >= 1024 ** 3) {
        return (
            `${(
                numeric / 1024 ** 3
            ).toFixed(2)} GB`
        );
    }

    if (numeric >= 1024 ** 2) {
        return (
            `${(
                numeric / 1024 ** 2
            ).toFixed(1)} MB`
        );
    }

    if (numeric >= 1024) {
        return (
            `${(
                numeric / 1024
            ).toFixed(1)} KB`
        );
    }

    return `${numeric} B`;
}


function resetProgressClasses(container) {
    container.classList.remove(
        "is-indeterminate",
        "is-complete",
        "is-error"
    );
}


function showModelProgress({
    label,
    detail = "Preparing model…",
    percentage = null,
    indeterminate = false
}) {
    const elements =
        getProgressElements();

    if (
        !elements.container
        || !elements.fill
    ) {
        return;
    }

    if (progressHideTimeout) {
        clearTimeout(
            progressHideTimeout
        );

        progressHideTimeout = null;
    }

    elements.container.hidden = false;

    resetProgressClasses(
        elements.container
    );

    if (elements.label) {
        elements.label.textContent =
            label;
    }

    if (elements.detail) {
        elements.detail.textContent =
            detail;
    }

    const numericPercentage =
        Number(percentage);

    if (
        indeterminate
        || !Number.isFinite(
            numericPercentage
        )
    ) {
        elements.container.classList.add(
            "is-indeterminate"
        );

        elements.fill.style.width = "";

        if (elements.percent) {
            elements.percent.textContent =
                "…";
        }

        if (elements.track) {
            elements.track.removeAttribute(
                "aria-valuenow"
            );
        }

        return;
    }

    const normalizedPercentage =
        Math.max(
            0,
            Math.min(
                100,
                numericPercentage
            )
        );

    elements.fill.style.width =
        `${normalizedPercentage}%`;

    if (elements.percent) {
        elements.percent.textContent =
            `${Math.round(
                normalizedPercentage
            )}%`;
    }

    if (elements.track) {
        elements.track.setAttribute(
            "aria-valuenow",
            String(
                Math.round(
                    normalizedPercentage
                )
            )
        );
    }
}


function completeModelProgress(
    label,
    detail
) {
    const elements =
        getProgressElements();

    if (
        !elements.container
        || !elements.fill
    ) {
        return;
    }

    if (progressHideTimeout) {
        clearTimeout(
            progressHideTimeout
        );
    }

    elements.container.hidden = false;

    resetProgressClasses(
        elements.container
    );

    elements.container.classList.add(
        "is-complete"
    );

    elements.fill.style.width = "100%";

    if (elements.label) {
        elements.label.textContent =
            label;
    }

    if (elements.percent) {
        elements.percent.textContent =
            "100%";
    }

    if (elements.detail) {
        elements.detail.textContent =
            detail;
    }

    if (elements.track) {
        elements.track.setAttribute(
            "aria-valuenow",
            "100"
        );
    }

    progressHideTimeout = setTimeout(
        () => {
            elements.container.hidden =
                true;

            resetProgressClasses(
                elements.container
            );

            elements.fill.style.width =
                "0%";
        },
        1800
    );
}


function failModelProgress(
    label,
    error
) {
    const elements =
        getProgressElements();

    if (!elements.container) {
        return;
    }

    if (progressHideTimeout) {
        clearTimeout(
            progressHideTimeout
        );

        progressHideTimeout = null;
    }

    elements.container.hidden = false;

    resetProgressClasses(
        elements.container
    );

    elements.container.classList.add(
        "is-error"
    );

    if (elements.label) {
        elements.label.textContent =
            label;
    }

    if (elements.percent) {
        elements.percent.textContent =
            "ERROR";
    }

    if (elements.detail) {
        elements.detail.textContent =
            error instanceof Error
                ? error.message
                : String(error);
    }
}


function createProgressCallback(stageLabel) {
    return progress => {
        if (!progress) {
            return;
        }

        const status = String(
            progress.status || ""
        ).toLowerCase();

        const fileName =
            progress.file
            || progress.name
            || "model files";

        let percentage =
            Number(
                progress.progress
            );

        const loaded =
            Number(progress.loaded);

        const total =
            Number(progress.total);

        if (
            !Number.isFinite(percentage)
            && Number.isFinite(loaded)
            && Number.isFinite(total)
            && total > 0
        ) {
            percentage = (
                loaded / total * 100
            );
        }

        let detail = fileName;

        const loadedText =
            formatDownloadSize(
                loaded
            );

        const totalText =
            formatDownloadSize(
                total
            );

        if (
            loadedText
            && totalText
            && total > 0
        ) {
            detail += (
                ` / ${loadedText}`
                + ` of ${totalText}`
            );
        }

        if (
            status === "progress"
            || status === "download"
        ) {
            showModelProgress({
                label: stageLabel,
                detail,
                percentage
            });

            return;
        }

        if (
            status === "initiate"
            || status === "ready"
        ) {
            showModelProgress({
                label: stageLabel,
                detail,
                indeterminate: true
            });
        }
    };
}

/* ==========================================================
   TEXT ENCODER LOADING
   ========================================================== */

async function loadTextEncoder() {
    if (
        tokenizer
        && textModel
    ) {
        return;
    }

    if (textModelPromise) {
        return textModelPromise;
    }

    if (!searchMetadata) {
        throw new Error(
            "Search metadata has not been loaded."
        );
    }

    const status =
        document.getElementById(
            "semantic-status"
        );

    const modelId =
        searchMetadata
            .model
            .browser_model;

    textModelPromise = (
        async () => {
            try {
                showModelProgress({
                    label:
                        "TEXT SEARCH / TOKENIZER",
                    detail:
                        "Preparing the CLIP tokenizer…",
                    indeterminate: true
                });

                tokenizer =
                    await AutoTokenizer
                        .from_pretrained(
                            modelId,
                            {
                                progress_callback:
                                    createProgressCallback(
                                        "TEXT SEARCH / TOKENIZER"
                                    )
                            }
                        );

                showModelProgress({
                    label:
                        "TEXT SEARCH / CLIP ENCODER",
                    detail:
                        "Preparing the text encoder…",
                    indeterminate: true
                });

                textModel =
                    await CLIPTextModelWithProjection
                        .from_pretrained(
                            modelId,
                            {
                                dtype:
                                    MODEL_DTYPE,

                                progress_callback:
                                    createProgressCallback(
                                        "TEXT SEARCH / CLIP ENCODER"
                                    )
                            }
                        );

                completeModelProgress(
                    "TEXT SEARCH / READY",
                    "CLIP text encoder loaded."
                );

                if (status) {
                    status.textContent =
                        "CLIP text encoder ready.";
                }

            } catch (error) {
                tokenizer = null;
                textModel = null;

                failModelProgress(
                    "TEXT SEARCH / FAILED",
                    error
                );

                throw error;
            }
        }
    )();

    try {
        await textModelPromise;

    } finally {
        textModelPromise = null;
    }
}

/* ==========================================================
   VISION ENCODER LOADING
   ========================================================== */

async function loadVisionEncoder() {
    if (
        processor
        && visionModel
    ) {
        return;
    }

    if (visionModelPromise) {
        return visionModelPromise;
    }

    if (!searchMetadata) {
        throw new Error(
            "Search metadata has not been loaded."
        );
    }

    const status =
        document.getElementById(
            "semantic-status"
        );

    const modelId =
        searchMetadata
            .model
            .browser_model;

    visionModelPromise = (
        async () => {
            try {
                showModelProgress({
                    label:
                        "IMAGE SEARCH / PROCESSOR",
                    detail:
                        "Preparing the CLIP image processor…",
                    indeterminate: true
                });

                processor =
                    await AutoProcessor
                        .from_pretrained(
                            modelId,
                            {
                                progress_callback:
                                    createProgressCallback(
                                        "IMAGE SEARCH / PROCESSOR"
                                    )
                            }
                        );

                showModelProgress({
                    label:
                        "IMAGE SEARCH / CLIP ENCODER",
                    detail:
                        "Preparing the vision encoder…",
                    indeterminate: true
                });

                visionModel =
                    await CLIPVisionModelWithProjection
                        .from_pretrained(
                            modelId,
                            {
                                dtype:
                                    MODEL_DTYPE,

                                progress_callback:
                                    createProgressCallback(
                                        "IMAGE SEARCH / CLIP ENCODER"
                                    )
                            }
                        );

                completeModelProgress(
                    "IMAGE SEARCH / READY",
                    "CLIP vision encoder loaded."
                );

                if (status) {
                    status.textContent =
                        "CLIP vision encoder ready.";
                }

            } catch (error) {
                processor = null;
                visionModel = null;

                failModelProgress(
                    "IMAGE SEARCH / FAILED",
                    error
                );

                throw error;
            }
        }
    )();

    try {
        await visionModelPromise;

    } finally {
        visionModelPromise = null;
    }
}

/* ==========================================================
   EMBEDDING EXTRACTION
   ========================================================== */

function extractProjectedEmbedding(
    output,
    preferredNames
) {
    if (!output) {
        throw new Error(
            "CLIP returned an empty output."
        );
    }

    /*
     * In some Transformers.js versions, the model may
     * return a Tensor directly.
     */
    if (
        output.data
        && output.dims
    ) {
        return Float32Array.from(
            output.data
        );
    }

    /*
     * Preferred projected outputs:
     * text_embeds or image_embeds.
     */
    for (
        const name
        of preferredNames
    ) {
        const tensor =
            output[name];

        if (
            tensor
            && tensor.data
        ) {
            return Float32Array.from(
                tensor.data
            );
        }
    }

    /*
     * Defensive fallback for output objects containing
     * exactly one projected two-dimensional tensor.
     */
    for (
        const value
        of Object.values(output)
    ) {
        if (
            value
            && value.data
            && value.dims
            && value.dims.length === 2
        ) {
            return Float32Array.from(
                value.data
            );
        }
    }

    throw new Error(
        "Could not locate a projected "
        + "CLIP embedding."
    );
}


function validateQueryDimension(vector) {
    const expectedDimension =
        Number(
            searchMetadata
                .model
                .dimension
        );

    if (
        vector.length
        !== expectedDimension
    ) {
        throw new Error(
            `Query dimension ${vector.length} `
            + "does not match the CLIP "
            + `index dimension ${expectedDimension}.`
        );
    }
}


/* ==========================================================
   TEXT EMBEDDING
   ========================================================== */

async function encodeText(query) {
    await loadTextEncoder();

    const inputs =
        tokenizer(
            [query],
            {
                padding: true,
                truncation: true
            }
        );

    const output =
        await textModel(
            inputs
        );

    const vector =
        extractProjectedEmbedding(
            output,
            [
                "text_embeds",
                "pooler_output"
            ]
        );

    normalizeVector(
        vector
    );

    validateQueryDimension(
        vector
    );

    return vector;
}


/* ==========================================================
   IMAGE LOADING
   ========================================================== */

async function fileToRawImage(file) {
    const objectURL =
        URL.createObjectURL(
            file
        );

    try {
        if (
            typeof RawImage.fromURL
            === "function"
        ) {
            return await RawImage.fromURL(
                objectURL
            );
        }

        if (
            typeof RawImage.read
            === "function"
        ) {
            return await RawImage.read(
                objectURL
            );
        }

        throw new Error(
            "This Transformers.js version "
            + "cannot decode uploaded images."
        );

    } finally {
        URL.revokeObjectURL(
            objectURL
        );
    }
}


/* ==========================================================
   IMAGE EMBEDDING
   ========================================================== */

async function encodeImage(file) {
    await loadVisionEncoder();

    const rawImage =
        await fileToRawImage(
            file
        );

    const inputs =
        await processor(
            rawImage
        );

    /*
     * This is the separate vision encoder. Unlike the
     * complete multimodal model, it does not require
     * input_ids.
     */
    const output =
        await visionModel(
            inputs
        );

    const vector =
        extractProjectedEmbedding(
            output,
            [
                "image_embeds",
                "pooler_output"
            ]
        );

    normalizeVector(
        vector
    );

    validateQueryDimension(
        vector
    );

    return vector;
}


async function encodeImageFromUrl(url) {
    await loadVisionEncoder();

    const rawImage =
        await RawImage.fromURL(
            url
        );

    const inputs =
        await processor(
            rawImage
        );

    const output =
        await visionModel(
            inputs
        );

    const vector =
        extractProjectedEmbedding(
            output,
            [
                "image_embeds",
                "pooler_output"
            ]
        );

    normalizeVector(
        vector
    );

    validateQueryDimension(
        vector
    );

    return vector;
}


/* ==========================================================
   FIND SIMILAR IMAGES
   ========================================================== */

function createSimilarSvgButton(
    parent,
    x,
    y,
    onClick
) {
    const namespace =
        "http://www.w3.org/2000/svg";

    const group =
        document.createElementNS(
            namespace,
            "g"
        );

    group.setAttribute(
        "class",
        "similar-svg-button"
    );

    const background =
        document.createElementNS(
            namespace,
            "circle"
        );

    background.setAttribute(
        "cx",
        String(x)
    );

    background.setAttribute(
        "cy",
        String(y)
    );

    background.setAttribute(
        "r",
        "10"
    );

    background.setAttribute(
        "class",
        "similar-svg-bg"
    );

    const magnifier =
        document.createElementNS(
            namespace,
            "g"
        );

    magnifier.setAttribute(
        "class",
        "similar-svg-mag"
    );

    const lens =
        document.createElementNS(
            namespace,
            "circle"
        );

    lens.setAttribute(
        "cx",
        String(x - 1.6)
    );

    lens.setAttribute(
        "cy",
        String(y - 1.6)
    );

    lens.setAttribute(
        "r",
        "4.4"
    );

    lens.setAttribute(
        "class",
        "similar-svg-lens"
    );

    const handle =
        document.createElementNS(
            namespace,
            "line"
        );

    handle.setAttribute(
        "x1",
        String(x + 1.4)
    );

    handle.setAttribute(
        "y1",
        String(y + 1.4)
    );

    handle.setAttribute(
        "x2",
        String(x + 5.4)
    );

    handle.setAttribute(
        "y2",
        String(y + 5.4)
    );

    handle.setAttribute(
        "class",
        "similar-svg-handle"
    );

    magnifier.appendChild(
        lens
    );

    magnifier.appendChild(
        handle
    );

    group.appendChild(
        background
    );

    group.appendChild(
        magnifier
    );

    parent.appendChild(
        group
    );

    group.addEventListener(
        "pointerdown",
        event => {
            event.stopPropagation();
        }
    );

    group.addEventListener(
        "pointerup",
        event => {
            event.stopPropagation();
        }
    );

    group.addEventListener(
        "click",
        event => {
            event.stopPropagation();
            event.preventDefault();

            onClick();
        }
    );

    group.addEventListener(
        "keydown",
        event => {
            if (
                event.key === "Enter"
                || event.key === " "
            ) {
                event.preventDefault();
                event.stopPropagation();

                onClick();
            }
        }
    );

    return group;
}


async function findSimilarImages(url) {
    if (!url) {
        return;
    }

    const modeElement =
        document.getElementById(
            "semantic-mode"
        );

    if (modeElement) {
        modeElement.value =
            "image";
    }

    updateSearchMode();

    const preview =
        document.getElementById(
            "semantic-upload-preview"
        );

    if (preview) {
        preview.innerHTML = `
            <img
                src="${escapeHtml(
                    url
                )}"
                alt="Similar image query"
            />
        `;
    }

    if (
        selectedImageURL
        && selectedImageURL.startsWith(
            "blob:"
        )
    ) {
        URL.revokeObjectURL(
            selectedImageURL
        );
    }

    selectedImageFile = null;
    selectedImageURL = url;

    try {
        await runImageSearch();
    } finally {
        scrollToSemanticResults();
    }
}


function scrollToSemanticResults() {
    const section =
        document.getElementById(
            "semantic-search"
        );

    if (!section) {
        return;
    }

    const scrollAndFlash = () => {
        section.scrollIntoView({
            behavior: "smooth",
            block: "start"
        });

        const intro =
            section.querySelector(
                ".semantic-search-intro"
            )
            || section;

        intro.classList.add(
            "semantic-results-flash"
        );

        window.setTimeout(
            () => {
                intro.classList.remove(
                    "semantic-results-flash"
                );
            },
            1400
        );
    };

    if (section.classList.contains("show")) {
        scrollAndFlash();
        return;
    }

    const tabEl =
        document.getElementById(
            "tab-semantic-search"
        );

    if (!tabEl) {
        scrollAndFlash();
        return;
    }

    tabEl.addEventListener(
        "shown.bs.tab",
        scrollAndFlash,
        { once: true }
    );

    tabEl.click();
}

window.findSimilarImages =
    findSimilarImages;

window.createSimilarSvgButton =
    createSimilarSvgButton;

function normalizeVector(vector) {
    let norm = 0;

    for (
        let index = 0;
        index < vector.length;
        index += 1
    ) {
        norm += (
            vector[index]
            * vector[index]
        );
    }

    norm = Math.sqrt(
        norm
    );

    if (
        !Number.isFinite(norm)
        || norm === 0
    ) {
        throw new Error(
            "CLIP produced an invalid embedding."
        );
    }

    for (
        let index = 0;
        index < vector.length;
        index += 1
    ) {
        vector[index] /=
            norm;
    }

    return vector;
}


/* ==========================================================
   VECTOR SEARCH
   ========================================================== */

function computeFullRanking(
    queryVector
) {
    const dimension = Number(
        searchMetadata
            .model
            .dimension
    );

    validateQueryDimension(
        queryVector
    );

    const results = [];

    for (
        let itemIndex = 0;
        itemIndex
        < searchMetadata.items.length;
        itemIndex += 1
    ) {
        const item =
            searchMetadata.items[
                itemIndex
            ];

        const offset =
            itemIndex
            * dimension;

        let score = 0;

        /*
         * The image embeddings and query embedding are
         * L2-normalized. Their dot product is cosine
         * similarity.
         */
        for (
            let vectorIndex = 0;
            vectorIndex < dimension;
            vectorIndex += 1
        ) {
            score += (
                queryVector[
                    vectorIndex
                ]
                * searchEmbeddings[
                    offset
                    + vectorIndex
                ]
            );
        }

        results.push({
            ...item,
            item_index:
                itemIndex,
            semantic_score:
                score
        });
    }

    results.sort(
        (first, second) =>
            second.semantic_score
            - first.semantic_score
    );

    return results;
}


function searchIndex(
    queryVector,
    selectedYear,
    minimumConfidence,
    limit
) {
    let ranking =
        rankingCache.get(
            queryVector
        );

    if (!ranking) {
        ranking =
            computeFullRanking(
                queryVector
            );

        if (
            rankingCache.size >= 24
        ) {
            const oldest =
                rankingCache.keys()
                    .next().value;

            rankingCache.delete(
                oldest
            );
        }

        rankingCache.set(
            queryVector,
            ranking
        );
    }

    const results = [];

    for (const result of ranking) {
        if (
            selectedYear
            && String(
                result.year
            ) !== selectedYear
        ) {
            continue;
        }

        if (
            Number(
                result.confidence
            ) < minimumConfidence
        ) {
            continue;
        }

        results.push(result);

        if (
            results.length >= limit
        ) {
            break;
        }
    }

    return results;
}


/* ==========================================================
   CURRENT FILTERS
   ========================================================== */

function currentSearchOptions() {
    const yearElement =
        document.getElementById(
            "semantic-year"
        );

    const confidenceElement =
        document.getElementById(
            "semantic-confidence"
        );

    const limitElement =
        document.getElementById(
            "semantic-limit"
        );

    return {
        year:
            yearElement
                ? yearElement.value
                : "",

        confidence:
            confidenceElement
                ? Number(
                    confidenceElement.value
                ) / 100
                : 0,

        limit:
            limitElement
                ? Number(
                    limitElement.value
                )
                : 24
    };
}


/* ==========================================================
   TEXT SEARCH
   ========================================================== */

async function runTextSearch() {
    const input =
        document.getElementById(
            "semantic-query"
        );

    const button =
        document.getElementById(
            "semantic-text-search-button"
        );

    const status =
        document.getElementById(
            "semantic-status"
        );

    const query =
        input
            ? input.value.trim()
            : "";

    if (!query) {
        status.textContent =
            "Enter a text description.";

        return;
    }

    button.disabled = true;

    try {
        status.textContent =
            `Encoding “${query}” with CLIP…`;

        const vector =
            await encodeText(
                query
            );

        const options =
            currentSearchOptions();

        const results =
            searchIndex(
                vector,
                options.year,
                options.confidence,
                options.limit
            );

        renderResults(
            results,
            `TEXT / ${query}`
        );

        status.textContent =
            `${results.length} results for “${query}”. `
            + "Search covers the public image sample.";

    } catch (error) {
        console.error(
            "CLIP text search failed:",
            error
        );

        status.textContent =
            "Text search failed. "
            + "See the browser console.";

    } finally {
        button.disabled = false;
    }
}


/* ==========================================================
   IMAGE SEARCH
   ========================================================== */

async function runImageSearch() {
    const button =
        document.getElementById(
            "semantic-image-search-button"
        );

    const status =
        document.getElementById(
            "semantic-status"
        );

    if (
        !selectedImageFile
        && !selectedImageURL
    ) {
        status.textContent =
            "Select an image first.";

        return;
    }

    button.disabled = true;

    try {
        status.textContent =
            "Encoding the selected image "
            + "with CLIP…";

        const vector =
            selectedImageFile
                ? await encodeImage(
                    selectedImageFile
                )
                : await encodeImageFromUrl(
                    selectedImageURL
                );

        const options =
            currentSearchOptions();

        const results =
            searchIndex(
                vector,
                options.year,
                options.confidence,
                options.limit
            );

        renderResults(
            results,
            "IMAGE EXAMPLE"
        );

        status.textContent =
            `${results.length} visually similar `
            + "images found.";

    } catch (error) {
        console.error(
            "CLIP image search failed:",
            error
        );

        status.textContent =
            "Image search failed. "
            + "See the browser console.";

    } finally {
        button.disabled = false;
    }
}


/* ==========================================================
   RESULT RENDERING
   ========================================================== */

function renderResults(
    results,
    queryLabel
) {
    const container =
        document.getElementById(
            "semantic-results"
        );

    if (!container) {
        return;
    }

    container.innerHTML = "";

    if (
        results.length === 0
    ) {
        container.innerHTML = `
            <div class="archive-note">
                No results match the selected filters.
            </div>
        `;

        return;
    }

    const fragment =
        document.createDocumentFragment();

    results.forEach(
        (
            item,
            resultIndex
        ) => {
            const card =
                document.createElement(
                    "div"
                );

            card.className =
                "semantic-result-card";

            card.setAttribute(
                "role",
                "button"
            );

            card.setAttribute(
                "tabindex",
                "0"
            );

            const score =
                Number(
                    item.semantic_score
                );

            const confidence =
                Number(
                    item.confidence
                );

            const imageMarkup =
                item.crop_image
                    ? `
                        <img
                            src="${escapeHtml(
                                item.crop_image
                            )}"
                            loading="lazy"
                            alt="CLIP search result"
                        />
                    `
                    : `
                        <span class="missing-image">
                            IMAGE UNAVAILABLE
                        </span>
                    `;

            const membershipMap = {
                both: {
                    label: "SHARED",
                    className: "semantic-membership-both"
                },
                a: {
                    label: "ONLY A",
                    className: "semantic-membership-a"
                },
                b: {
                    label: "ONLY B",
                    className: "semantic-membership-b"
                }
            };

            const membership =
                membershipMap[item.membership];

            const membershipMarkup =
                membership
                    ? `
                        <span class="semantic-membership ${membership.className}">
                            ${membership.label}
                        </span>
                    `
                    : "";

            const similarButton =
                item.crop_image
                    ? `
                        <button
                            class="similar-image-button"
                            type="button"
                            title="Find similar images"
                            aria-label="Find similar images"
                        >
                            <svg
                                viewBox="0 0 16 16"
                                aria-hidden="true"
                            >
                                <circle
                                    cx="7"
                                    cy="7"
                                    r="4.6"
                                ></circle>
                                <line
                                    x1="10.4"
                                    y1="10.4"
                                    x2="14.5"
                                    y2="14.5"
                                ></line>
                            </svg>
                        </button>
                    `
                    : "";

            card.innerHTML = `
                <span class="semantic-rank">
                    ${String(
                        resultIndex + 1
                    ).padStart(2, "0")}
                </span>

                <span class="semantic-image">
                    ${imageMarkup}
                    ${similarButton}
                </span>

                <span class="semantic-result-data">

                    ${membershipMarkup}

                    <strong>
                        SIMILARITY
                        ${Number.isFinite(score)
                            ? score.toFixed(3)
                            : "N/A"}
                    </strong>

                    <small>
                        ${escapeHtml(
                            item.year
                        )}
                        /
                        PAGE
                        ${escapeHtml(
                            item.page
                        )}
                    </small>

                    <small>
                        YOLO
                        ${Number.isFinite(confidence)
                            ? (
                                confidence * 100
                            ).toFixed(1)
                            : "N/A"}%
                    </small>

                </span>
            `;

            card.addEventListener(
                "click",
                () => {
                    openModal(
                        item,
                        queryLabel
                    );
                }
            );

            card.addEventListener(
                "keydown",
                event => {
                    if (
                        event.key === "Enter"
                        || event.key === " "
                    ) {
                        event.preventDefault();

                        openModal(
                            item,
                            queryLabel
                        );
                    }
                }
            );

            const similar =
                card.querySelector(
                    ".similar-image-button"
                );

            if (similar) {
                similar.addEventListener(
                    "click",
                    event => {
                        event.stopPropagation();

                        findSimilarImages(
                            item.crop_image
                        );
                    }
                );

                similar.addEventListener(
                    "keydown",
                    event => {
                        if (
                            event.key === "Enter"
                            || event.key === " "
                        ) {
                            event.stopPropagation();
                        }
                    }
                );
            }

            fragment.appendChild(
                card
            );
        }
    );

    container.appendChild(
        fragment
    );
}


/* ==========================================================
   SEARCH MODE
   ========================================================== */

function updateSearchMode() {
    const modeElement =
        document.getElementById(
            "semantic-mode"
        );

    const textPanel =
        document.getElementById(
            "semantic-text-panel"
        );

    const imagePanel =
        document.getElementById(
            "semantic-image-panel"
        );

    const graphPanel =
        document.getElementById(
            "semantic-graph-panel"
        );

    const graphWrap =
        document.getElementById(
            "semantic-graph-wrap"
        );

    const limitControl =
        document.getElementById(
            "semantic-limit-control"
        );

    if (
        !modeElement
        || !textPanel
        || !imagePanel
    ) {
        return;
    }

    const mode =
        modeElement.value;

    textPanel.hidden =
        mode !== "text";

    imagePanel.hidden =
        mode !== "image";

    if (graphPanel) {
        graphPanel.hidden =
            mode !== "graph";
    }

    if (limitControl) {
        limitControl.hidden =
            mode === "graph";
    }

    if (
        graphWrap
        && mode !== "graph"
    ) {
        graphWrap.hidden = true;
    }
}


/* ==========================================================
   IMAGE SELECTION
   ========================================================== */

function handleImageSelection(event) {
    const file =
        event.target.files?.[0];

    const preview =
        document.getElementById(
            "semantic-upload-preview"
        );

    const button =
        document.getElementById(
            "semantic-image-search-button"
        );

    if (selectedImageURL) {
        URL.revokeObjectURL(
            selectedImageURL
        );

        selectedImageURL = null;
    }

    selectedImageFile = null;

    if (button) {
        button.disabled = true;
    }

    if (!file) {
        if (preview) {
            preview.innerHTML =
                "<span>No image selected</span>";
        }

        return;
    }

    if (
        !file.type.startsWith(
            "image/"
        )
    ) {
        if (preview) {
            preview.innerHTML =
                "<span>The selected file is not an image.</span>";
        }

        return;
    }

    selectedImageFile =
        file;

    selectedImageURL =
        URL.createObjectURL(
            file
        );

    if (preview) {
        preview.innerHTML = `
            <img
                src="${selectedImageURL}"
                alt="Selected query image"
            />

            <span>
                ${escapeHtml(
                    file.name
                )}
            </span>
        `;
    }

    if (button) {
        button.disabled = false;
    }
}


/* ==========================================================
   YEAR FILTER
   ========================================================== */

function populateYears(items) {
    const select =
        document.getElementById(
            "semantic-year"
        );

    if (!select) {
        return;
    }

    const years = [
        ...new Set(
            items
                .map(
                    item => item.year
                )
                .filter(
                    value =>
                        value !== null
                        && value !== undefined
                        && value !== ""
                )
        )
    ].sort(
        (first, second) =>
            Number(first)
            - Number(second)
    );

    for (const year of years) {
        const option =
            document.createElement(
                "option"
            );

        option.value =
            String(year);

        option.textContent =
            String(year);

        select.appendChild(
            option
        );
    }
}


/* ==========================================================
   CONTROL EVENTS
   ========================================================== */

function connectControls() {
    const modeSelect =
        document.getElementById(
            "semantic-mode"
        );

    const queryInput =
        document.getElementById(
            "semantic-query"
        );

    const textButton =
        document.getElementById(
            "semantic-text-search-button"
        );

    const imageInput =
        document.getElementById(
            "semantic-image-input"
        );

    const imageButton =
        document.getElementById(
            "semantic-image-search-button"
        );

    const confidenceInput =
        document.getElementById(
            "semantic-confidence"
        );

    const confidenceValue =
        document.getElementById(
            "semantic-confidence-value"
        );

    if (modeSelect) {
        modeSelect.addEventListener(
            "change",
            updateSearchMode
        );
    }

    if (textButton) {
        textButton.addEventListener(
            "click",
            runTextSearch
        );
    }

    if (queryInput) {
        queryInput.addEventListener(
            "keydown",
            event => {
                if (
                    event.key === "Enter"
                ) {
                    runTextSearch();
                }
            }
        );
    }

    if (imageInput) {
        imageInput.addEventListener(
            "change",
            handleImageSelection
        );
    }

    if (imageButton) {
        imageButton.addEventListener(
            "click",
            runImageSearch
        );
    }

    if (
        confidenceInput
        && confidenceValue
    ) {
        confidenceInput.addEventListener(
            "input",
            () => {
                confidenceValue.textContent =
                    `${confidenceInput.value}%`;
            }
        );
    }

    document
        .querySelectorAll(
            ".semantic-example"
        )
        .forEach(
            button => {
                button.addEventListener(
                    "click",
                    () => {
                        if (queryInput) {
                            queryInput.value =
                                button.dataset.query
                                || "";
                        }

                        if (modeSelect) {
                            modeSelect.value =
                                "text";

                            updateSearchMode();
                        }

                        runTextSearch();
                    }
                );
            }
        );

    const graphButton =
        document.getElementById(
            "semantic-graph-search-button"
        );

    const graphQueryA =
        document.getElementById(
            "semantic-query-a"
        );

    const graphQueryB =
        document.getElementById(
            "semantic-query-b"
        );

    const graphLimit =
        document.getElementById(
            "semantic-graph-limit"
        );

    const graphLinks =
        document.getElementById(
            "semantic-graph-show-query-links"
        );

    if (graphButton) {
        graphButton.addEventListener(
            "click",
            runIntersectionGraph
        );
    }

    for (const graphInput of [
        graphQueryA,
        graphQueryB
    ]) {
        if (graphInput) {
            graphInput.addEventListener(
                "keydown",
                event => {
                    if (
                        event.key === "Enter"
                    ) {
                        runIntersectionGraph();
                    }
                }
            );
        }
    }

    for (const graphControl of [
        graphLimit,
        graphLinks
    ]) {
        if (graphControl) {
            graphControl.addEventListener(
                "change",
                refreshGraphFromControls
            );
        }
    }

    const yearElement =
        document.getElementById(
            "semantic-year"
        );

    if (yearElement) {
        yearElement.addEventListener(
            "change",
            refreshGraphFromControls
        );
    }

    const confidenceElement =
        document.getElementById(
            "semantic-confidence"
        );

    if (confidenceElement) {
        confidenceElement.addEventListener(
            "change",
            refreshGraphFromControls
        );
    }

    const modal =
        document.getElementById(
            "semantic-modal"
        );

    const modalClose =
        document.getElementById(
            "semantic-modal-close"
        );

    if (modalClose) {
        modalClose.addEventListener(
            "click",
            closeModal
        );
    }

    if (modal) {
        modal.addEventListener(
            "click",
            event => {
                if (
                    event.target
                    === modal
                ) {
                    closeModal();
                }
            }
        );
    }

    document.addEventListener(
        "keydown",
        event => {
            if (
                event.key === "Escape"
            ) {
                closeModal();
            }
        }
    );

    updateSearchMode();
}


/* ==========================================================
   RESULT MODAL
   ========================================================== */

function openModal(
    item,
    queryLabel
) {
    const modal =
        document.getElementById(
            "semantic-modal"
        );

    const body =
        document.getElementById(
            "semantic-modal-body"
        );

    if (
        !modal
        || !body
    ) {
        return;
    }

    const cropImage =
        item.crop_image
            ? `
                <img
                    class="modal-image"
                    src="${escapeHtml(
                        item.crop_image
                    )}"
                    alt="Semantic search result"
                />
            `
            : `
                <div class="missing-image">
                    IMAGE UNAVAILABLE
                </div>
            `;

    const pageImage =
        item.page_image
            ? `
                <img
                    class="modal-image"
                    src="${escapeHtml(
                        item.page_image
                    )}"
                    alt="Newspaper page"
                />
            `
            : `
                <div class="missing-image">
                    PAGE PREVIEW UNAVAILABLE
                </div>
            `;

    const score =
        Number(
            item.semantic_score
        );

    const confidence =
        Number(
            item.confidence
        );

    body.innerHTML = `
        <div class="modal-grid">

            <section>

                <h3>
                    Search result
                </h3>

                ${cropImage}

            </section>

            <section>

                <h3>
                    Page context
                </h3>

                ${pageImage}

            </section>

        </div>

        <section class="metadata-panel">

            <h3>
                ${escapeHtml(
                    queryLabel
                )}
            </h3>

            <dl>

                <dt>Similarity</dt>

                <dd>
                    ${Number.isFinite(score)
                        ? score.toFixed(4)
                        : "N/A"}
                </dd>

                <dt>YOLO confidence</dt>

                <dd>
                    ${Number.isFinite(confidence)
                        ? (
                            confidence * 100
                        ).toFixed(1)
                        : "N/A"}%
                </dd>

                <dt>Year</dt>

                <dd>
                    ${escapeHtml(
                        item.year
                    )}
                </dd>

                <dt>Page</dt>

                <dd>
                    ${escapeHtml(
                        item.page
                    )}
                </dd>

                <dt>Journal</dt>

                <dd>
                    ${escapeHtml(
                        item.journal
                    )}
                </dd>

                <dt>Element ID</dt>

                <dd>
                    <code>
                        ${escapeHtml(
                            item.id
                        )}
                    </code>
                </dd>

            </dl>

            ${
                item.url
                    ? `
                        <a
                            class="source-link"
                            href="${escapeHtml(
                                item.url
                            )}"
                            target="_blank"
                            rel="noopener noreferrer"
                        >
                            Open original source
                        </a>
                    `
                    : ""
            }

        </section>
    `;

    modal.classList.add(
        "visible"
    );

    modal.setAttribute(
        "aria-hidden",
        "false"
    );

    document.body.style.overflow =
        "hidden";
}


function closeModal() {
    const modal =
        document.getElementById(
            "semantic-modal"
        );

    if (!modal) {
        return;
    }

    modal.classList.remove(
        "visible"
    );

    modal.setAttribute(
        "aria-hidden",
        "true"
    );

    document.body.style.overflow =
        "";
}


/* ==========================================================
   HTML ESCAPING
   ========================================================== */

function escapeHtml(value) {
    return String(
        value ?? ""
    )
        .replaceAll(
            "&",
            "&amp;"
        )
        .replaceAll(
            "<",
            "&lt;"
        )
        .replaceAll(
            ">",
            "&gt;"
        )
        .replaceAll(
            '"',
            "&quot;"
        )
        .replaceAll(
            "'",
            "&#039;"
        );
}


/* ==========================================================
   QUERY INTERSECTION GRAPH
   ========================================================== */

const GRAPH_VIEWBOX_WIDTH = 1000;
const GRAPH_VIEWBOX_HEIGHT = 620;
const GRAPH_NODE_SIZE = 46;
const GRAPH_IMAGE_SIZE = 38;
const GRAPH_HITBOX_SIZE = 48;

let currentGraphState = null;
let graphRefreshTimeout = null;


function getItemEmbedding(itemIndex) {
    const dimension = Number(
        searchMetadata.model.dimension
    );

    const offset = itemIndex * dimension;

    return searchEmbeddings.subarray(
        offset,
        offset + dimension
    );
}


function vectorDotProduct(first, second) {
    let dot = 0;

    for (
        let index = 0;
        index < first.length;
        index += 1
    ) {
        dot += first[index] * second[index];
    }

    return dot;
}


function hashString(value) {
    let hash = 0;

    for (
        let index = 0;
        index < value.length;
        index += 1
    ) {
        hash = (
            hash * 31
            + value.charCodeAt(index)
        ) >>> 0;
    }

    return hash;
}


function currentGraphOptions() {
    const limitElement =
        document.getElementById(
            "semantic-graph-limit"
        );

    const linksElement =
        document.getElementById(
            "semantic-graph-show-query-links"
        );

    const searchOptions =
        currentSearchOptions();

    return {
        limit:
            limitElement
                ? Number(
                    limitElement.value
                )
                : 24,

        showQueryLinks:
            linksElement
                ? linksElement.checked
                : true,

        year:
            searchOptions.year,

        confidence:
            searchOptions.confidence
    };
}


function queryLabel(text) {
    const cleaned = String(
        text ?? ""
    ).trim();

    if (cleaned.length <= 24) {
        return cleaned;
    }

    return `${cleaned.slice(0, 23)}…`;
}


function buildIntersectionGraph(
    queryA,
    queryB,
    options
) {
    const resultsA =
        searchIndex(
            queryA.vector,
            options.year,
            options.confidence,
            options.limit
        );

    const resultsB =
        searchIndex(
            queryB.vector,
            options.year,
            options.confidence,
            options.limit
        );

    const idsB = new Set(
        resultsB.map(
            result => result.id
        )
    );

    const intersection =
        resultsA.filter(
            result => idsB.has(result.id)
        );

    const intersectionIds =
        new Set(
            intersection.map(
                result => result.id
            )
        );

    const unionById = new Map();

    for (const result of resultsA) {
        unionById.set(
            result.id,
            {
                item: result,
                membership:
                    intersectionIds.has(
                        result.id
                    )
                        ? "both"
                        : "a"
            }
        );
    }

    for (const result of resultsB) {
        const existing =
            unionById.get(result.id);

        if (existing) {
            existing.membership =
                "both";
        } else {
            unionById.set(
                result.id,
                {
                    item: result,
                    membership: "b"
                }
            );
        }
    }

    const nodes = [];

    nodes.push({
        id: "query-A",
        kind: "query",
        label: queryLabel(queryA.text),
        fixed: true
    });

    nodes.push({
        id: "query-B",
        kind: "query",
        label: queryLabel(queryB.text),
        fixed: true
    });

    for (const [id, info] of unionById) {
        nodes.push({
            id,
            kind: "image",
            item: info.item,
            membership: info.membership
        });
    }

    const edges = [];

    if (options.showQueryLinks) {
        for (const result of resultsA) {
            edges.push({
                source: "query-A",
                target: result.id,
                kind: "query-a"
            });
        }

        for (const result of resultsB) {
            edges.push({
                source: "query-B",
                target: result.id,
                kind: "query-b"
            });
        }
    }

    const membershipOrder = {
        both: 0,
        a: 1,
        b: 2
    };

    const union =
        Array.from(
            unionById.values()
        ).map(info => ({
            ...info.item,
            membership: info.membership
        })).sort(
            (first, second) => {
                const membershipDiff =
                    membershipOrder[first.membership]
                    - membershipOrder[second.membership];

                if (membershipDiff !== 0) {
                    return membershipDiff;
                }

                return (
                    Number(second.semantic_score)
                    - Number(first.semantic_score)
                );
            }
        );

    return {
        nodes,
        edges,
        intersection,
        union
    };
}


function membershipAnchor(
    node,
    width,
    height
) {
    if (node.membership === "a") {
        return {
            x: width * 0.33,
            y: height * 0.5
        };
    }

    if (node.membership === "b") {
        return {
            x: width * 0.67,
            y: height * 0.5
        };
    }

    return {
        x: width * 0.5,
        y: height * 0.5
    };
}


function simulateGraphLayout(
    nodes,
    edges,
    width,
    height
) {
    const movable =
        nodes.filter(
            node => !node.fixed
        );

    const count = Math.max(
        movable.length,
        1
    );

    const area = width * height;

    const ideal =
        Math.sqrt(
            area / count
        ) * 0.85;

    for (const node of movable) {
        const seed = hashString(
            node.id
        );

        const seedX =
            (seed % 1000) / 1000;

        const seedY =
            ((seed >>> 10) % 1000) / 1000;

        node.x =
            width / 2
            + (seedX - 0.5) * width * 0.4;

        node.y =
            height / 2
            + (seedY - 0.5) * height * 0.4;

        node.vx = 0;
        node.vy = 0;
    }

    for (const node of nodes) {
        if (node.id === "query-A") {
            node.x = 120;
            node.y = height / 2;
        } else if (node.id === "query-B") {
            node.x = width - 120;
            node.y = height / 2;
        }
    }

    const nodeById = new Map();

    for (const node of nodes) {
        nodeById.set(node.id, node);
    }

    const iterations = Math.max(
        70,
        Math.min(
            280,
            Math.round(
                280 - movable.length * 1.5
            )
        )
    );

    for (
        let iteration = 0;
        iteration < iterations;
        iteration += 1
    ) {
        const cooling =
            1 - iteration / iterations;

        for (
            let first = 0;
            first < movable.length;
            first += 1
        ) {
            for (
                let second = first + 1;
                second < movable.length;
                second += 1
            ) {
                const a = movable[first];
                const b = movable[second];

                let dx = a.x - b.x;
                let dy = a.y - b.y;

                let distance = Math.sqrt(
                    dx * dx + dy * dy
                );

                if (distance < 1) {
                    dx = 0.5;
                    dy = 0.5;
                    distance = 1;
                }

                const force =
                    ideal * ideal / distance
                    * 0.015 * cooling;

                const fx =
                    dx / distance * force;

                const fy =
                    dy / distance * force;

                a.vx += fx;
                a.vy += fy;
                b.vx -= fx;
                b.vy -= fy;
            }
        }

        for (const edge of edges) {
            const source =
                nodeById.get(edge.source);

            const target =
                nodeById.get(edge.target);

            if (!source || !target) {
                continue;
            }

            let restLength = ideal;

            if (
                edge.kind === "query-a"
                || edge.kind === "query-b"
            ) {
                restLength = ideal * 0.6;
            } else {
                restLength = ideal * 0.85;
            }

            let dx = target.x - source.x;
            let dy = target.y - source.y;

            const distance = Math.sqrt(
                dx * dx + dy * dy
            ) || 0.01;

            const force =
                (distance - restLength)
                * 0.01 * cooling;

            const fx = dx / distance * force;
            const fy = dy / distance * force;

            if (!source.fixed) {
                source.vx += fx;
                source.vy += fy;
            }

            if (!target.fixed) {
                target.vx -= fx;
                target.vy -= fy;
            }
        }

        for (const node of movable) {
            const anchor =
                membershipAnchor(
                    node,
                    width,
                    height
                );

            node.vx +=
                (anchor.x - node.x)
                * 0.10 * cooling;

            node.vy +=
                (anchor.y - node.y)
                * 0.08 * cooling;

            node.vx *= 0.86;
            node.vy *= 0.86;

            node.x += node.vx;
            node.y += node.vy;

            const margin =
                GRAPH_HITBOX_SIZE / 2 + 8;

            if (node.x < margin) {
                node.x = margin;
            } else if (
                node.x > width - margin
            ) {
                node.x = width - margin;
            }

            if (node.y < margin) {
                node.y = margin;
            } else if (
                node.y > height - margin
            ) {
                node.y = height - margin;
            }
        }
    }

    return nodes;
}


function showGraphTooltip(node) {
    const tooltip =
        document.getElementById(
            "semantic-graph-tooltip"
        );

    if (!tooltip) {
        return;
    }

    const item = node.item;

    const confidence = Number(
        item.confidence
    );

    tooltip.innerHTML = `
        <strong>
            ${escapeHtml(
                item.year
            )}
            / PAGE ${escapeHtml(
                item.page
            )}
        </strong>

        <span>
            SIMILARITY ${Number(
                item.semantic_score
            ).toFixed(3)}
        </span>

        <span>
            YOLO ${Number.isFinite(confidence)
                ? (confidence * 100).toFixed(1)
                : "N/A"}%
        </span>
    `;

    tooltip.classList.add("visible");
}


function hideGraphTooltip() {
    const tooltip =
        document.getElementById(
            "semantic-graph-tooltip"
        );

    if (!tooltip) {
        return;
    }

    tooltip.classList.remove("visible");
}


let graphDragState = null;


function svgPointFromEvent(svg, event) {
    const point =
        svg.createSVGPoint();

    point.x = event.clientX;
    point.y = event.clientY;

    const matrix =
        svg.getScreenCTM();

    if (!matrix) {
        return null;
    }

    return point.matrixTransform(
        matrix.inverse()
    );
}


function updateGraphNodePosition(refs, node) {
    const half =
        GRAPH_NODE_SIZE / 2;

    const imageHalf =
        GRAPH_IMAGE_SIZE / 2;

    const hitboxHalf =
        GRAPH_HITBOX_SIZE / 2;

    refs.frame.setAttribute(
        "x",
        String(node.x - half)
    );
    refs.frame.setAttribute(
        "y",
        String(node.y - half)
    );
    refs.image.setAttribute(
        "x",
        String(node.x - imageHalf)
    );
    refs.image.setAttribute(
        "y",
        String(node.y - imageHalf)
    );
    refs.hitbox.setAttribute(
        "x",
        String(node.x - hitboxHalf)
    );
    refs.hitbox.setAttribute(
        "y",
        String(node.y - hitboxHalf)
    );
}


function updateGraphNodeEdges(id, node, svgEdges) {
    for (const edge of svgEdges) {
        if (edge.source === id) {
            edge.element.setAttribute(
                "x1",
                String(node.x)
            );
            edge.element.setAttribute(
                "y1",
                String(node.y)
            );
        } else if (
            edge.target === id
        ) {
            edge.element.setAttribute(
                "x2",
                String(node.x)
            );
            edge.element.setAttribute(
                "y2",
                String(node.y)
            );
        }
    }
}


function renderIntersectionGraph(graph) {
    const svg =
        document.getElementById(
            "semantic-graph"
        );

    if (!svg) {
        return;
    }

    const namespace =
        "http://www.w3.org/2000/svg";

    svg.innerHTML = "";

    const nodes =
        simulateGraphLayout(
            graph.nodes,
            graph.edges,
            GRAPH_VIEWBOX_WIDTH,
            GRAPH_VIEWBOX_HEIGHT
        );

    const nodeById = new Map();

    for (const node of nodes) {
        nodeById.set(node.id, node);
    }

    const svgNodes = new Map();

    const svgEdges = [];

    const fragment =
        document.createDocumentFragment();

    const background =
        document.createElementNS(
            namespace,
            "rect"
        );

    background.setAttribute(
        "x",
        "0"
    );
    background.setAttribute(
        "y",
        "0"
    );
    background.setAttribute(
        "width",
        String(GRAPH_VIEWBOX_WIDTH)
    );
    background.setAttribute(
        "height",
        String(GRAPH_VIEWBOX_HEIGHT)
    );
    background.setAttribute(
        "class",
        "semantic-graph-background"
    );

    fragment.appendChild(background);

    for (const edge of graph.edges) {
        const source =
            nodeById.get(edge.source);

        const target =
            nodeById.get(edge.target);

        if (!source || !target) {
            continue;
        }

        const line =
            document.createElementNS(
                namespace,
                "line"
            );

        line.setAttribute(
            "x1",
            String(source.x)
        );
        line.setAttribute(
            "y1",
            String(source.y)
        );
        line.setAttribute(
            "x2",
            String(target.x)
        );
        line.setAttribute(
            "y2",
            String(target.y)
        );

        line.setAttribute(
            "class",
            "semantic-graph-edge "
            + `semantic-graph-edge-${edge.kind}`
        );

        svgEdges.push({
            element: line,
            source: edge.source,
            target: edge.target
        });

        fragment.appendChild(line);
    }

    for (const node of nodes) {
        if (node.kind !== "query") {
            continue;
        }

        const group =
            document.createElementNS(
                namespace,
                "g"
            );

        const queryClass =
            node.id === "query-A"
                ? "semantic-graph-query-node-a"
                : "semantic-graph-query-node-b";

        group.setAttribute(
            "class",
            "semantic-graph-query-node "
            + queryClass
        );

        const circle =
            document.createElementNS(
                namespace,
                "circle"
            );

        circle.setAttribute(
            "cx",
            String(node.x)
        );
        circle.setAttribute(
            "cy",
            String(node.y)
        );
        circle.setAttribute(
            "r",
            "22"
        );

        group.appendChild(circle);

        const label =
            document.createElementNS(
                namespace,
                "text"
            );

        label.setAttribute(
            "x",
            String(node.x)
        );
        label.setAttribute(
            "y",
            String(node.y + 44)
        );
        label.setAttribute(
            "class",
            "semantic-graph-query-label"
        );

        label.textContent =
            node.label || "QUERY";

        group.appendChild(label);

        fragment.appendChild(group);
    }

    for (const node of nodes) {
        if (node.kind !== "image") {
            continue;
        }

        const group =
            document.createElementNS(
                namespace,
                "g"
            );

        group.setAttribute(
            "class",
            "semantic-graph-node"
        );
        group.setAttribute(
            "tabindex",
            "0"
        );
        group.setAttribute(
            "role",
            "button"
        );

        const half =
            GRAPH_NODE_SIZE / 2;

        const frame =
            document.createElementNS(
                namespace,
                "rect"
            );

        frame.setAttribute(
            "x",
            String(node.x - half)
        );
        frame.setAttribute(
            "y",
            String(node.y - half)
        );
        frame.setAttribute(
            "width",
            String(GRAPH_NODE_SIZE)
        );
        frame.setAttribute(
            "height",
            String(GRAPH_NODE_SIZE)
        );
        const membershipClass =
            node.membership === "both"
                ? "semantic-graph-node-frame-both"
                : node.membership === "a"
                    ? "semantic-graph-node-frame-a"
                    : node.membership === "b"
                        ? "semantic-graph-node-frame-b"
                        : "semantic-graph-node-frame";

        frame.setAttribute(
            "class",
            membershipClass
        );

        group.appendChild(frame);

        const imageHalf =
            GRAPH_IMAGE_SIZE / 2;

        const image =
            document.createElementNS(
                namespace,
                "image"
            );

        image.setAttribute(
            "x",
            String(node.x - imageHalf)
        );
        image.setAttribute(
            "y",
            String(node.y - imageHalf)
        );
        image.setAttribute(
            "width",
            String(GRAPH_IMAGE_SIZE)
        );
        image.setAttribute(
            "height",
            String(GRAPH_IMAGE_SIZE)
        );
        image.setAttribute(
            "preserveAspectRatio",
            "xMidYMid slice"
        );

        if (node.item.crop_image) {
            image.setAttribute(
                "href",
                node.item.crop_image
            );
            image.setAttributeNS(
                "http://www.w3.org/1999/xlink",
                "href",
                node.item.crop_image
            );
        }

        group.appendChild(image);

        const hitboxHalf =
            GRAPH_HITBOX_SIZE / 2;

        const hitbox =
            document.createElementNS(
                namespace,
                "rect"
            );

        hitbox.setAttribute(
            "x",
            String(node.x - hitboxHalf)
        );
        hitbox.setAttribute(
            "y",
            String(node.y - hitboxHalf)
        );
        hitbox.setAttribute(
            "width",
            String(GRAPH_HITBOX_SIZE)
        );
        hitbox.setAttribute(
            "height",
            String(GRAPH_HITBOX_SIZE)
        );
        hitbox.setAttribute(
            "class",
            "semantic-graph-hitbox"
        );

        group.appendChild(hitbox);

        if (node.item.crop_image) {
            createSimilarSvgButton(
                group,
                node.x + half - 11,
                node.y - half + 11,
                () => findSimilarImages(
                    node.item.crop_image
                )
            );
        }

        svgNodes.set(node.id, {
            frame,
            image,
            hitbox
        });

        group.addEventListener(
            "mouseenter",
            () => showGraphTooltip(node)
        );
        group.addEventListener(
            "mouseleave",
            hideGraphTooltip
        );
        group.addEventListener(
            "focus",
            () => showGraphTooltip(node)
        );
        group.addEventListener(
            "blur",
            hideGraphTooltip
        );
        group.addEventListener(
            "pointerdown",
            event => {
                event.preventDefault();

                const point =
                    svgPointFromEvent(
                        svg,
                        event
                    );

                if (!point) {
                    return;
                }

                graphDragState = {
                    id: node.id,
                    startX: point.x,
                    startY: point.y,
                    moved: false
                };

                group.setPointerCapture(
                    event.pointerId
                );
            }
        );
        group.addEventListener(
            "pointermove",
            event => {
                if (
                    !graphDragState
                    || graphDragState.id !== node.id
                ) {
                    return;
                }

                const point =
                    svgPointFromEvent(
                        svg,
                        event
                    );

                if (!point) {
                    return;
                }

                const delta =
                    Math.hypot(
                        point.x - graphDragState.startX,
                        point.y - graphDragState.startY
                    );

                if (
                    !graphDragState.moved
                    && delta < 4
                ) {
                    return;
                }

                graphDragState.moved = true;

                node.x = point.x;
                node.y = point.y;

                const refs =
                    svgNodes.get(node.id);

                if (refs) {
                    updateGraphNodePosition(
                        refs,
                        node
                    );

                    updateGraphNodeEdges(
                        node.id,
                        node,
                        svgEdges
                    );
                }
            }
        );
        group.addEventListener(
            "pointerup",
            event => {
                const wasMoved =
                    graphDragState
                    && graphDragState.moved;

                graphDragState = null;

                if (!wasMoved) {
                    openModal(
                        node.item,
                        "INTERSECTION / QUERY GRAPH"
                    );
                }
            }
        );
        group.addEventListener(
            "pointercancel",
            () => {
                graphDragState = null;
            }
        );
        group.addEventListener(
            "keydown",
            event => {
                if (
                    event.key === "Enter"
                    || event.key === " "
                ) {
                    event.preventDefault();

                    openModal(
                        node.item,
                        "INTERSECTION / QUERY GRAPH"
                    );
                }
            }
        );

        fragment.appendChild(group);
    }

    svg.appendChild(fragment);
}


function updateGraphSummary(graph, options) {
    const summary =
        document.getElementById(
            "semantic-graph-summary"
        );

    if (!summary) {
        return;
    }

    const shared =
        graph.intersection.length;

    let onlyA = 0;
    let onlyB = 0;

    for (const node of graph.nodes) {
        if (node.kind !== "image") {
            continue;
        }

        if (node.membership === "a") {
            onlyA += 1;
        } else if (
            node.membership === "b"
        ) {
            onlyB += 1;
        }
    }

    if (shared === 0) {
        summary.textContent =
            `QUERY A ∩ QUERY B — no shared images. `
            + `Only in A: ${onlyA} · only in B: ${onlyB} `
            + `(of ${options.limit} per query). `
            + `Try two related queries (e.g. "church" and `
            + `"cathedral", or "train" and "railway").`;

        return;
    }

    summary.textContent =
        `QUERY A ∩ QUERY B — ${shared} shared · `
        + `only in A: ${onlyA} · only in B: ${onlyB} `
        + `(of ${options.limit} per query).`;
}


async function runIntersectionGraph() {
    const queryAElement =
        document.getElementById(
            "semantic-query-a"
        );

    const queryBElement =
        document.getElementById(
            "semantic-query-b"
        );

    const button =
        document.getElementById(
            "semantic-graph-search-button"
        );

    const status =
        document.getElementById(
            "semantic-status"
        );

    const queryA =
        queryAElement
            ? queryAElement.value.trim()
            : "";

    const queryB =
        queryBElement
            ? queryBElement.value.trim()
            : "";

    if (!queryA || !queryB) {
        status.textContent =
            "Enter both query A and query B.";

        return;
    }

    if (button) {
        button.disabled = true;
    }

    try {
        status.textContent =
            "Encoding both queries with CLIP…";

        const [
            vectorA,
            vectorB
        ] = await Promise.all([
            encodeText(queryA),
            encodeText(queryB)
        ]);

        const options =
            currentGraphOptions();

        const graph =
            buildIntersectionGraph(
                {
                    vector: vectorA,
                    text: queryA
                },
                {
                    vector: vectorB,
                    text: queryB
                },
                options
            );

        currentGraphState = {
            queryA: {
                vector: vectorA,
                text: queryA
            },
            queryB: {
                vector: vectorB,
                text: queryB
            },
            options,
            graph
        };

        const graphWrap =
            document.getElementById(
                "semantic-graph-wrap"
            );

        if (graphWrap) {
            graphWrap.hidden = false;
        }

        renderIntersectionGraph(
            graph
        );

        updateGraphSummary(
            graph,
            options
        );

        renderResults(
            graph.union,
            `UNION / ${queryA} ∩ ${queryB}`
        );

        status.textContent =
            `${graph.union.length} images — `
            + `${graph.intersection.length} shared `
            + `between “${queryA}” and “${queryB}”.`;

    } catch (error) {
        console.error(
            "CLIP intersection failed:",
            error
        );

        status.textContent =
            "Intersection failed. "
            + "See the browser console.";

    } finally {
        if (button) {
            button.disabled = false;
        }
    }
}


function refreshGraphFromControls() {
    if (!currentGraphState) {
        return;
    }

    const options =
        currentGraphOptions();

    const graph =
        buildIntersectionGraph(
            currentGraphState.queryA,
            currentGraphState.queryB,
            options
        );

    currentGraphState.options =
        options;

    currentGraphState.graph =
        graph;

    renderIntersectionGraph(
        graph
    );

    updateGraphSummary(
        graph,
        options
    );

    renderResults(
        graph.union,
        `UNION / ${
            currentGraphState.queryA.text
        } ∩ ${
            currentGraphState.queryB.text
        }`
    );
}


function scheduleGraphRefresh() {
    if (graphRefreshTimeout) {
        clearTimeout(
            graphRefreshTimeout
        );
    }

    graphRefreshTimeout = setTimeout(
        refreshGraphFromControls,
        120
    );
}


/* ==========================================================
   START
   ========================================================== */

if (
    document.readyState
    === "loading"
) {
    document.addEventListener(
        "DOMContentLoaded",
        initializeSemanticSearch
    );

} else {
    initializeSemanticSearch();
}