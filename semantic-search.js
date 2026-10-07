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


/* ==========================================================
   VECTOR NORMALIZATION
   ========================================================== */

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

function searchIndex(
    queryVector,
    selectedYear,
    minimumConfidence,
    limit
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

        if (
            selectedYear
            && String(
                item.year
            ) !== selectedYear
        ) {
            continue;
        }

        if (
            Number(
                item.confidence
            ) < minimumConfidence
        ) {
            continue;
        }

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
            semantic_score:
                score
        });
    }

    results.sort(
        (first, second) =>
            second.semantic_score
            - first.semantic_score
    );

    return results.slice(
        0,
        limit
    );
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

    if (!selectedImageFile) {
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
            await encodeImage(
                selectedImageFile
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
                    "button"
                );

            card.type =
                "button";

            card.className =
                "semantic-result-card";

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

            card.innerHTML = `
                <span class="semantic-rank">
                    ${String(
                        resultIndex + 1
                    ).padStart(2, "0")}
                </span>

                <span class="semantic-image">
                    ${imageMarkup}
                </span>

                <span class="semantic-result-data">

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