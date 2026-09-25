document.addEventListener(
    "DOMContentLoaded",
    async () => {
        const response = await fetch(
            "dashboard_data.json"
        );

        if (!response.ok) {
            console.error(
                "Could not load dashboard_data.json"
            );
            return;
        }

        const dashboard = await response.json();

        initializeGallery(
            dashboard.gallery || []
        );

        initializeSpatialHeatmap(
            dashboard.pages || [],
            dashboard.spatial_detections || []
        );
    }
);


function uniqueSorted(values) {
    return [
        ...new Set(
            values.filter(
                value =>
                    value !== null
                    && value !== undefined
                    && value !== ""
            )
        )
    ].sort((a, b) => {
        const numberA = Number(a);
        const numberB = Number(b);

        if (
            Number.isFinite(numberA)
            && Number.isFinite(numberB)
        ) {
            return numberA - numberB;
        }

        return String(a).localeCompare(
            String(b)
        );
    });
}


function escapeHtml(value) {
    return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}


function formatPercent(value, digits = 1) {
    const numeric = Number(value);

    if (!Number.isFinite(numeric)) {
        return "0%";
    }

    return `${(numeric * 100).toFixed(digits)}%`;
}


function initializeGallery(originalData) {
    const gallery = document.getElementById(
        "visual-gallery"
    );

    if (!gallery) {
        return;
    }

    const status = document.getElementById(
        "gallery-status"
    );

    const searchInput = document.getElementById(
        "gallery-search"
    );

    const classSelect = document.getElementById(
        "gallery-class"
    );

    const yearSelect = document.getElementById(
        "gallery-year"
    );

    const confidenceInput =
        document.getElementById(
            "gallery-confidence"
        );

    const confidenceValue =
        document.getElementById(
            "gallery-confidence-value"
        );

    const sortSelect = document.getElementById(
        "gallery-sort"
    );

    const modal = document.getElementById(
        "gallery-modal"
    );

    const modalBody = document.getElementById(
        "gallery-modal-body"
    );

    const modalClose = document.getElementById(
        "gallery-modal-close"
    );

    const classes = uniqueSorted(
        originalData.map(
            item => item.class
        )
    );

    const years = uniqueSorted(
        originalData.map(
            item => item.year
        )
    );

    for (const className of classes) {
        const option =
            document.createElement("option");

        option.value = className;
        option.textContent = className;

        classSelect.appendChild(option);
    }

    for (const year of years) {
        const option =
            document.createElement("option");

        option.value = year;
        option.textContent = year;

        yearSelect.appendChild(option);
    }

    function filteredData() {
        const search = searchInput.value
            .trim()
            .toLowerCase();

        const selectedClass =
            classSelect.value;

        const selectedYear =
            yearSelect.value;

        const minimumConfidence =
            Number(confidenceInput.value) / 100;

        const filtered = originalData.filter(
            item => {
                const searchable = [
                    item.id,
                    item.journal,
                    item.year,
                    item.page,
                    item.issue,
                    item.bib,
                    item.class
                ]
                    .join(" ")
                    .toLowerCase();

                const matchesSearch =
                    !search
                    || searchable.includes(search);

                const matchesClass =
                    !selectedClass
                    || item.class === selectedClass;

                const matchesYear =
                    !selectedYear
                    || String(item.year)
                        === selectedYear;

                const matchesConfidence =
                    Number(item.confidence)
                    >= minimumConfidence;

                return (
                    matchesSearch
                    && matchesClass
                    && matchesYear
                    && matchesConfidence
                );
            }
        );

        const sortMode = sortSelect.value;

        filtered.sort((a, b) => {
            if (
                sortMode
                === "confidence-asc"
            ) {
                return (
                    Number(a.confidence)
                    - Number(b.confidence)
                );
            }

            if (
                sortMode
                === "year-asc"
            ) {
                return (
                    Number(a.year)
                    - Number(b.year)
                );
            }

            if (
                sortMode
                === "year-desc"
            ) {
                return (
                    Number(b.year)
                    - Number(a.year)
                );
            }

            return (
                Number(b.confidence)
                - Number(a.confidence)
            );
        });

        return filtered;
    }

    function openModal(item) {
        const confidence = formatPercent(
            item.confidence
        );

        const crop = item.crop_image
            ? `
                <img
                    class="modal-image"
                    src="${escapeHtml(
                        item.crop_image
                    )}"
                    alt="Extracted visual element"
                />
            `
            : `
                <div class="missing-image">
                    Crop unavailable
                </div>
            `;

        const page = item.page_image
            ? `
                <img
                    class="modal-image"
                    src="${escapeHtml(
                        item.page_image
                    )}"
                    alt="Annotated newspaper page"
                />
            `
            : `
                <div class="missing-image">
                    Page preview unavailable
                </div>
            `;

        modalBody.innerHTML = `
            <div class="modal-grid">
                <section>
                    <h3>Extracted element</h3>
                    ${crop}
                </section>

                <section>
                    <h3>Page context</h3>
                    ${page}
                </section>
            </div>

            <section class="metadata-panel">
                <h3>
                    ${escapeHtml(item.class)}
                </h3>

                <dl>
                    <dt>Confidence</dt>
                    <dd>${confidence}</dd>

                    <dt>Journal</dt>
                    <dd>
                        ${escapeHtml(
                            item.journal
                        )}
                    </dd>

                    <dt>Year</dt>
                    <dd>
                        ${escapeHtml(
                            item.year
                        )}
                    </dd>

                    <dt>Issue</dt>
                    <dd>
                        ${escapeHtml(
                            item.issue
                        )}
                    </dd>

                    <dt>Page</dt>
                    <dd>
                        ${escapeHtml(
                            item.page
                        )}
                    </dd>

                    <dt>Page detections</dt>
                    <dd>
                        ${escapeHtml(
                            item.detections_on_page
                        )}
                    </dd>

                    <dt>Bounding box</dt>
                    <dd>
                        <code>
                        ${escapeHtml(
                            JSON.stringify(
                                item.bbox
                            )
                        )}
                        </code>
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

        modal.classList.add("visible");
        modal.setAttribute(
            "aria-hidden",
            "false"
        );
    }

    function closeModal() {
        modal.classList.remove("visible");

        modal.setAttribute(
            "aria-hidden",
            "true"
        );
    }

    function render() {
        const data = filteredData();

        confidenceValue.textContent =
            `${confidenceInput.value}%`;

        status.textContent =
            `${data.length.toLocaleString()} visual elements`;

        gallery.innerHTML = "";

        const fragment =
            document.createDocumentFragment();

        for (const item of data) {
            const confidence = formatPercent(
                item.confidence
            );

            const card =
                document.createElement("article");

            card.className = "gallery-card";
            card.tabIndex = 0;

            const image = item.crop_image
                ? `
                    <img
                        src="${escapeHtml(
                            item.crop_image
                        )}"
                        loading="lazy"
                        alt="${escapeHtml(
                            item.class
                        )}"
                    />
                `
                : `
                    <div class="missing-image">
                        Image unavailable
                    </div>
                `;

            card.innerHTML = `
                <div class="gallery-image-container">
                    ${image}

                    <span class="confidence-badge">
                        ${confidence}
                    </span>
                </div>

                <div class="gallery-card-body">
                    <div class="class-label">
                        ${escapeHtml(
                            item.class
                        )}
                    </div>

                    <div class="gallery-metadata">
                        ${escapeHtml(
                            item.journal
                        )}
                        · ${escapeHtml(
                            item.year
                        )}
                        · p. ${escapeHtml(
                            item.page
                        )}
                    </div>
                </div>
            `;

            card.addEventListener(
                "click",
                () => openModal(item)
            );

            card.addEventListener(
                "keydown",
                event => {
                    if (
                        event.key === "Enter"
                        || event.key === " "
                    ) {
                        event.preventDefault();
                        openModal(item);
                    }
                }
            );

            fragment.appendChild(card);
        }

        gallery.appendChild(fragment);
    }

    const controls = [
        searchInput,
        classSelect,
        yearSelect,
        confidenceInput,
        sortSelect
    ];

    for (const control of controls) {
        control.addEventListener(
            "input",
            render
        );

        control.addEventListener(
            "change",
            render
        );
    }

    modalClose.addEventListener(
        "click",
        closeModal
    );

    modal.addEventListener(
        "click",
        event => {
            if (event.target === modal) {
                closeModal();
            }
        }
    );

    document.addEventListener(
        "keydown",
        event => {
            if (event.key === "Escape") {
                closeModal();
            }
        }
    );

    render();
}


function initializeSpatialHeatmap(
    allPages,
    allDetections
) {
    const canvas = document.getElementById(
        "spatial-heatmap"
    );

    if (!canvas) {
        return;
    }

    const yearSelect = document.getElementById(
        "heatmap-year"
    );

    const classSelect = document.getElementById(
        "heatmap-class"
    );

    const confidenceInput =
        document.getElementById(
            "heatmap-confidence"
        );

    const confidenceValue =
        document.getElementById(
            "heatmap-confidence-value"
        );

    const metricsContainer =
        document.getElementById(
            "spatial-metrics"
        );

    const years = uniqueSorted(
        allPages.map(
            item => item.year
        )
    );

    const classes = uniqueSorted(
        allDetections.map(
            item => item.class
        )
    );

    for (const year of years) {
        const option =
            document.createElement("option");

        option.value = year;
        option.textContent = year;

        yearSelect.appendChild(option);
    }

    for (const className of classes) {
        const option =
            document.createElement("option");

        option.value = className;
        option.textContent = className;

        classSelect.appendChild(option);
    }

    /*
     * Every page is represented by a binary grid.
     * Bounding boxes mark grid cells as occupied.
     *
     * The final heatmap is the proportion of selected
     * pages in which each cell was occupied.
     */
    const gridColumns = 36;
    const gridRows = 54;

    function selectedData() {
        const year = yearSelect.value;
        const className = classSelect.value;

        const threshold =
            Number(confidenceInput.value) / 100;

        const pages = allPages.filter(
            page =>
                !year
                || String(page.year) === year
        );

        const pageIds = new Set(
            pages.map(
                page => page.page_id
            )
        );

        const detections =
            allDetections.filter(
                detection =>
                    pageIds.has(
                        detection.page_id
                    )
                    && (
                        !className
                        || detection.class
                            === className
                    )
                    && Number(
                        detection.confidence
                    ) >= threshold
            );

        return {
            pages,
            detections,
            threshold,
        };
    }

    function colorForValue(
        normalizedValue
    ) {
        const value = Math.max(
            0,
            Math.min(
                1,
                normalizedValue
            )
        );

        /*
         * Sequential blue-green-yellow palette.
         */
        const stops = [
            {
                position: 0,
                color: [247, 250, 252],
            },
            {
                position: 0.25,
                color: [204, 235, 230],
            },
            {
                position: 0.50,
                color: [60, 174, 163],
            },
            {
                position: 0.75,
                color: [32, 99, 155],
            },
            {
                position: 1,
                color: [237, 85, 59],
            },
        ];

        let left = stops[0];
        let right = stops[
            stops.length - 1
        ];

        for (
            let index = 0;
            index < stops.length - 1;
            index += 1
        ) {
            if (
                value >= stops[index].position
                && value
                    <= stops[index + 1].position
            ) {
                left = stops[index];
                right = stops[index + 1];
                break;
            }
        }

        const interval =
            right.position - left.position;

        const amount = interval > 0
            ? (
                value - left.position
            ) / interval
            : 0;

        const color = left.color.map(
            (channel, index) =>
                Math.round(
                    channel
                    + (
                        right.color[index]
                        - channel
                    )
                    * amount
                )
        );

        return `rgb(
            ${color[0]},
            ${color[1]},
            ${color[2]}
        )`;
    }

    function computeSpatialStatistics(
        pages,
        detections
    ) {
        const pageMasks = new Map();

        for (const page of pages) {
            pageMasks.set(
                page.page_id,
                new Uint8Array(
                    gridColumns
                    * gridRows
                )
            );
        }

        for (const detection of detections) {
            const mask = pageMasks.get(
                detection.page_id
            );

            if (!mask) {
                continue;
            }

            const [
                x1,
                y1,
                x2,
                y2
            ] = detection.bbox;

            const startColumn = Math.max(
                0,
                Math.min(
                    gridColumns - 1,
                    Math.floor(
                        x1 * gridColumns
                    )
                )
            );

            const endColumn = Math.max(
                startColumn,
                Math.min(
                    gridColumns - 1,
                    Math.ceil(
                        x2 * gridColumns
                    ) - 1
                )
            );

            const startRow = Math.max(
                0,
                Math.min(
                    gridRows - 1,
                    Math.floor(
                        y1 * gridRows
                    )
                )
            );

            const endRow = Math.max(
                startRow,
                Math.min(
                    gridRows - 1,
                    Math.ceil(
                        y2 * gridRows
                    ) - 1
                )
            );

            for (
                let row = startRow;
                row <= endRow;
                row += 1
            ) {
                for (
                    let column = startColumn;
                    column <= endColumn;
                    column += 1
                ) {
                    mask[
                        row * gridColumns
                        + column
                    ] = 1;
                }
            }
        }

        const aggregate = new Float64Array(
            gridColumns * gridRows
        );

        let totalCoverage = 0;
        let pagesWithDetections = 0;

        for (const mask of pageMasks.values()) {
            let occupiedCells = 0;

            for (
                let index = 0;
                index < mask.length;
                index += 1
            ) {
                if (mask[index]) {
                    aggregate[index] += 1;
                    occupiedCells += 1;
                }
            }

            const pageCoverage =
                occupiedCells
                / mask.length;

            totalCoverage += pageCoverage;

            if (occupiedCells > 0) {
                pagesWithDetections += 1;
            }
        }

        const pageCount = pages.length;

        if (pageCount > 0) {
            for (
                let index = 0;
                index < aggregate.length;
                index += 1
            ) {
                aggregate[index] /=
                    pageCount;
            }
        }

        const meanCoverage =
            pageCount > 0
            ? totalCoverage / pageCount
            : 0;

        const detectionRate =
            pageCount > 0
            ? pagesWithDetections / pageCount
            : 0;

        const meanDetections =
            pageCount > 0
            ? detections.length / pageCount
            : 0;

        return {
            aggregate,
            meanCoverage,
            detectionRate,
            meanDetections,
            pageCount,
            pagesWithDetections,
            detectionCount: detections.length,
        };
    }

    function drawHeatmap(
        aggregate
    ) {
        const context = canvas.getContext(
            "2d"
        );

        const displayWidth =
            canvas.clientWidth || 540;

        const displayHeight =
            canvas.clientHeight || 760;

        const pixelRatio =
            window.devicePixelRatio || 1;

        const expectedWidth = Math.round(
            displayWidth * pixelRatio
        );

        const expectedHeight = Math.round(
            displayHeight * pixelRatio
        );

        if (
            canvas.width !== expectedWidth
            || canvas.height !== expectedHeight
        ) {
            canvas.width = expectedWidth;
            canvas.height = expectedHeight;
        }

        context.setTransform(
            pixelRatio,
            0,
            0,
            pixelRatio,
            0,
            0
        );

        context.clearRect(
            0,
            0,
            displayWidth,
            displayHeight
        );

        context.fillStyle = "#ffffff";

        context.fillRect(
            0,
            0,
            displayWidth,
            displayHeight
        );

        const maximum = Math.max(
            ...aggregate,
            0
        );

        const cellWidth =
            displayWidth / gridColumns;

        const cellHeight =
            displayHeight / gridRows;

        for (
            let row = 0;
            row < gridRows;
            row += 1
        ) {
            for (
                let column = 0;
                column < gridColumns;
                column += 1
            ) {
                const value = aggregate[
                    row * gridColumns
                    + column
                ];

                const normalized =
                    maximum > 0
                    ? value / maximum
                    : 0;

                context.fillStyle =
                    colorForValue(normalized);

                context.fillRect(
                    column * cellWidth,
                    row * cellHeight,
                    Math.ceil(cellWidth) + 0.3,
                    Math.ceil(cellHeight) + 0.3
                );
            }
        }

        /*
         * Page border and subtle historical-paper texture.
         */
        context.strokeStyle =
            "rgba(23, 63, 95, 0.65)";

        context.lineWidth = 2;

        context.strokeRect(
            1,
            1,
            displayWidth - 2,
            displayHeight - 2
        );

        context.fillStyle =
            "rgba(23, 63, 95, 0.7)";

        context.font =
            "12px Inter, sans-serif";

        context.fillText(
            "Top of page",
            12,
            21
        );
    }

    function renderMetrics(statistics) {
        const metrics = [
            {
                title: "Selected pages",
                value: statistics.pageCount
                    .toLocaleString(),
                description: (
                    "Pages included in the "
                    + "selected year"
                ),
            },
            {
                title: "Pages with detections",
                value: formatPercent(
                    statistics.detectionRate
                ),
                description: (
                    `${statistics
                        .pagesWithDetections
                        .toLocaleString()} pages`
                ),
            },
            {
                title: "Mean page coverage",
                value: formatPercent(
                    statistics.meanCoverage
                ),
                description: (
                    "Average occupied page area"
                ),
            },
            {
                title: "Elements per page",
                value: statistics
                    .meanDetections
                    .toFixed(2),
                description: (
                    `${statistics
                        .detectionCount
                        .toLocaleString()} detections`
                ),
            },
        ];

        metricsContainer.innerHTML =
            metrics.map(
                metric => `
                    <article
                        class="spatial-metric-card"
                    >
                        <div
                            class="spatial-metric-title"
                        >
                            ${metric.title}
                        </div>

                        <div
                            class="spatial-metric-value"
                        >
                            ${metric.value}
                        </div>

                        <div
                            class="spatial-metric-description"
                        >
                            ${metric.description}
                        </div>
                    </article>
                `
            ).join("");
    }

    function render() {
        const {
            pages,
            detections
        } = selectedData();

        confidenceValue.textContent =
            `${confidenceInput.value}%`;

        const statistics =
            computeSpatialStatistics(
                pages,
                detections
            );

        drawHeatmap(
            statistics.aggregate
        );

        renderMetrics(
            statistics
        );
    }

    const controls = [
        yearSelect,
        classSelect,
        confidenceInput
    ];

    for (const control of controls) {
        control.addEventListener(
            "input",
            render
        );

        control.addEventListener(
            "change",
            render
        );
    }

    const resizeObserver =
        new ResizeObserver(() => {
            render();
        });

    resizeObserver.observe(canvas);

    render();
}