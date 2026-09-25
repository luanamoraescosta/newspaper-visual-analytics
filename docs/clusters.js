document.addEventListener(
    "DOMContentLoaded",
    async () => {
        const mapElement =
            document.getElementById(
                "cluster-map"
            );

        if (!mapElement) {
            return;
        }

        try {
            const response = await fetch(
                "clusters.json"
            );

            if (!response.ok) {
                throw new Error(
                    `HTTP ${response.status}`
                );
            }

            const clusterData =
                await response.json();

            initializeClusterExplorer(
                clusterData
            );

        } catch (error) {
            console.error(
                "Could not load clusters.json:",
                error
            );

            const status =
                document.getElementById(
                    "cluster-map-status"
                );

            if (status) {
                status.textContent =
                    "Cluster data could not be loaded.";
            }
        }
    }
);


/* ==========================================================
   GENERAL HELPERS
   ========================================================== */

function clusterUniqueSorted(values) {
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


function clusterEscapeHtml(value) {
    return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}


function clusterFormatPercent(
    value,
    digits = 1
) {
    const numeric = Number(value);

    if (!Number.isFinite(numeric)) {
        return "N/A";
    }

    return (
        `${(numeric * 100).toFixed(digits)}%`
    );
}


function clusterLabel(clusterId) {
    if (Number(clusterId) === -1) {
        return "Noise / Unassigned";
    }

    return `Cluster ${clusterId}`;
}


/* ==========================================================
   CLUSTER EXPLORER
   ========================================================== */

function initializeClusterExplorer(
    clusterData
) {
    const allItems =
        clusterData.items || [];

    const summary =
        clusterData.summary || {};

    const method =
        clusterData.method || {};


    /* ======================================================
       ELEMENTS
       ====================================================== */

    const summaryContainer =
        document.getElementById(
            "cluster-summary"
        );

    const map =
        document.getElementById(
            "cluster-map"
        );

    const mapStatus =
        document.getElementById(
            "cluster-map-status"
        );

    const tooltip =
        document.getElementById(
            "cluster-tooltip"
        );

    const renderMode =
        document.getElementById(
            "cluster-render-mode"
        );

    const colorMode =
        document.getElementById(
            "cluster-color-mode"
        );

    const clusterFilter =
        document.getElementById(
            "cluster-filter"
        );

    const classFilter =
        document.getElementById(
            "cluster-class-filter"
        );

    const yearFilter =
        document.getElementById(
            "cluster-year-filter"
        );

    const exampleCount =
        document.getElementById(
            "cluster-example-count"
        );

    const showNoise =
        document.getElementById(
            "cluster-show-noise"
        );

    const representativeOnly =
        document.getElementById(
            "cluster-representative-only"
        );

    const atlas =
        document.getElementById(
            "cluster-atlas"
        );

    const atlasStatus =
        document.getElementById(
            "cluster-atlas-status"
        );

    const modal =
        document.getElementById(
            "cluster-image-modal"
        );

    const modalBody =
        document.getElementById(
            "cluster-image-modal-body"
        );

    const modalClose =
        document.getElementById(
            "cluster-image-modal-close"
        );


    if (
        !map
        || !atlas
        || !summaryContainer
    ) {
        return;
    }


    /* ======================================================
       VALUES AND COLOR PALETTE
       ====================================================== */

    const clusterIds = clusterUniqueSorted(
        allItems.map(
            item => Number(
                item.cluster
            )
        )
    );

    const regularClusterIds =
        clusterIds.filter(
            clusterId =>
                clusterId >= 0
        );

    const classes = clusterUniqueSorted(
        allItems.map(
            item => item.class
        )
    );

    const years = clusterUniqueSorted(
        allItems.map(
            item => item.year
        )
    );

    /*
     * Restricted archival palette.
     * Colors repeat when the number of clusters
     * exceeds the size of the palette.
     */
    const palette = [
        "#5b14ff",
        "#7faf98",
        "#59afc6",
        "#d7c93c",
        "#c78d3d",
        "#d65a8b",
        "#315bce",
        "#4e7765",
        "#8f65d9",
        "#2699b5",
        "#a9981e",
        "#9b6130"
    ];

    const clusterColors =
        new Map();

    regularClusterIds.forEach(
        (clusterId, index) => {
            clusterColors.set(
                clusterId,
                palette[
                    index
                    % palette.length
                ]
            );
        }
    );

    clusterColors.set(
        -1,
        "#99958c"
    );


    /* ======================================================
       POPULATE FILTERS
       ====================================================== */

    if (clusterFilter) {
        for (
            const clusterId
            of clusterIds
        ) {
            const option =
                document.createElement(
                    "option"
                );

            option.value =
                String(clusterId);

            option.textContent =
                clusterLabel(clusterId);

            clusterFilter.appendChild(
                option
            );
        }
    }

    /*
     * This filter may be hidden in index.qmd because
     * the detector currently has only one class.
     * It is still populated for compatibility.
     */
    if (classFilter) {
        for (
            const className
            of classes
        ) {
            const option =
                document.createElement(
                    "option"
                );

            option.value =
                String(className);

            option.textContent =
                String(className);

            classFilter.appendChild(
                option
            );
        }
    }

    if (yearFilter) {
        for (const year of years) {
            const option =
                document.createElement(
                    "option"
                );

            option.value =
                String(year);

            option.textContent =
                String(year);

            yearFilter.appendChild(
                option
            );
        }
    }


    /* ======================================================
       SUMMARY
       ====================================================== */

    function renderSummary() {
        const itemCount =
            Number(
                summary.items
            ) || 0;

        const clusterCount =
            Number(
                summary.cluster_count
            ) || 0;

        const groupedItems =
            Number(
                summary.grouped_items
            ) || 0;

        const noiseItems =
            Number(
                summary.noise_items
            ) || 0;

        const silhouette =
            summary.silhouette_score;

        const groupedPercentage =
            itemCount > 0
                ? groupedItems / itemCount
                : 0;

        const noisePercentage =
            itemCount > 0
                ? noiseItems / itemCount
                : 0;

        const embeddingModel =
            method.embedding_model
            || "OpenCLIP";

        const cards = [
            {
                title: "Visual clusters",
                value: clusterCount
                    .toLocaleString(),
                detail: "HDBSCAN groups",
                className: "purple"
            },
            {
                title: "Indexed images",
                value: itemCount
                    .toLocaleString(),
                detail: embeddingModel,
                className: "green"
            },
            {
                title: "Images grouped",
                value:
                    clusterFormatPercent(
                        groupedPercentage
                    ),
                detail:
                    `${groupedItems
                        .toLocaleString()} images`,
                className: "blue"
            },
            {
                title: "Unassigned",
                value:
                    clusterFormatPercent(
                        noisePercentage
                    ),
                detail:
                    `${noiseItems
                        .toLocaleString()} images`,
                className: "orange"
            },
            {
                title: "Silhouette",
                value:
                    silhouette === null
                    || silhouette === undefined
                        ? "N/A"
                        : Number(
                            silhouette
                        ).toFixed(3),
                detail:
                    "Unassigned items excluded",
                className: "purple"
            }
        ];

        summaryContainer.innerHTML =
            cards.map(
                card => `
                    <article
                        class="
                            cluster-summary-card
                            ${card.className}
                        "
                    >
                        <div
                            class="cluster-summary-title"
                        >
                            ${clusterEscapeHtml(
                                card.title
                            )}
                        </div>

                        <div
                            class="cluster-summary-value"
                        >
                            ${clusterEscapeHtml(
                                card.value
                            )}
                        </div>

                        <div
                            class="cluster-summary-detail"
                        >
                            ${clusterEscapeHtml(
                                card.detail
                            )}
                        </div>
                    </article>
                `
            ).join("");
    }


    /* ======================================================
       FILTERED DATA
       ====================================================== */

    function getFilteredItems() {
        const selectedCluster =
            clusterFilter
                ? clusterFilter.value
                : "";

        const selectedClass =
            classFilter
                ? classFilter.value
                : "";

        const selectedYear =
            yearFilter
                ? yearFilter.value
                : "";

        return allItems.filter(
            item => {
                const matchesCluster =
                    selectedCluster === ""
                    || String(
                        item.cluster
                    ) === selectedCluster;

                const matchesClass =
                    selectedClass === ""
                    || String(
                        item.class
                    ) === selectedClass;

                const matchesYear =
                    selectedYear === ""
                    || String(
                        item.year
                    ) === selectedYear;

                return (
                    matchesCluster
                    && matchesClass
                    && matchesYear
                );
            }
        );
    }


    /* ======================================================
       COLORS
       ====================================================== */

    function interpolateColor(
        start,
        end,
        amount
    ) {
        const parseColor = color => [
            parseInt(
                color.slice(1, 3),
                16
            ),
            parseInt(
                color.slice(3, 5),
                16
            ),
            parseInt(
                color.slice(5, 7),
                16
            )
        ];

        const first =
            parseColor(start);

        const second =
            parseColor(end);

        const normalizedAmount =
            Math.max(
                0,
                Math.min(
                    1,
                    Number(amount) || 0
                )
            );

        const values = first.map(
            (value, index) =>
                Math.round(
                    value
                    + (
                        second[index]
                        - value
                    )
                    * normalizedAmount
                )
        );

        return (
            "#"
            + values.map(
                value =>
                    value
                        .toString(16)
                        .padStart(2, "0")
            ).join("")
        );
    }


    function categoricalColor(
        value,
        possibleValues
    ) {
        const index =
            possibleValues.indexOf(
                value
            );

        if (index === -1) {
            return "#99958c";
        }

        return palette[
            index % palette.length
        ];
    }


    function pointColor(item) {
        const mode =
            colorMode
                ? colorMode.value
                : "cluster";

        if (mode === "class") {
            return categoricalColor(
                item.class,
                classes
            );
        }

        if (mode === "year") {
            const validYears = years
                .map(Number)
                .filter(
                    Number.isFinite
                );

            const year =
                Number(item.year);

            if (
                !Number.isFinite(year)
                || validYears.length === 0
            ) {
                return "#99958c";
            }

            const minimum =
                Math.min(
                    ...validYears
                );

            const maximum =
                Math.max(
                    ...validYears
                );

            const amount =
                maximum === minimum
                    ? 0.5
                    : (
                        year - minimum
                    )
                    / (
                        maximum - minimum
                    );

            return interpolateColor(
                "#7faf98",
                "#5b14ff",
                amount
            );
        }

        if (mode === "confidence") {
            const confidence =
                Math.max(
                    0,
                    Math.min(
                        1,
                        Number(
                            item.confidence
                        ) || 0
                    )
                );

            return interpolateColor(
                "#d7c93c",
                "#5b14ff",
                confidence
            );
        }

        return (
            clusterColors.get(
                Number(
                    item.cluster
                )
            )
            || "#99958c"
        );
    }


    /* ======================================================
       MAP BOUNDS
       ====================================================== */

    function getBounds(items) {
        const validItems =
            items.filter(
                item =>
                    Number.isFinite(
                        Number(item.x)
                    )
                    && Number.isFinite(
                        Number(item.y)
                    )
            );

        if (
            validItems.length === 0
        ) {
            return {
                minX: 0,
                maxX: 1,
                minY: 0,
                maxY: 1
            };
        }

        const xValues =
            validItems.map(
                item => Number(item.x)
            );

        const yValues =
            validItems.map(
                item => Number(item.y)
            );

        let minX =
            Math.min(...xValues);

        let maxX =
            Math.max(...xValues);

        let minY =
            Math.min(...yValues);

        let maxY =
            Math.max(...yValues);

        if (minX === maxX) {
            minX -= 1;
            maxX += 1;
        }

        if (minY === maxY) {
            minY -= 1;
            maxY += 1;
        }

        /*
         * Adds a small margin around the projection.
         */
        const xPadding =
            (maxX - minX) * 0.04;

        const yPadding =
            (maxY - minY) * 0.04;

        return {
            minX: minX - xPadding,
            maxX: maxX + xPadding,
            minY: minY - yPadding,
            maxY: maxY + yPadding
        };
    }


    /* ======================================================
       MAP TOOLTIP
       ====================================================== */

    function showTooltip(
        event,
        item
    ) {
        if (!tooltip) {
            return;
        }

        const imagePreview =
            item.crop_image
                ? `
                    <img
                        src="${clusterEscapeHtml(
                            item.crop_image
                        )}"
                        alt=""
                        style="
                            width: 88px;
                            height: 72px;
                            object-fit: contain;
                            margin-bottom: 0.4rem;
                            background: #fffdf7;
                            border: 1px solid #fffdf7;
                        "
                    />
                `
                : "";

        tooltip.innerHTML = `
            ${imagePreview}

            <strong>
                ${clusterEscapeHtml(
                    clusterLabel(
                        item.cluster
                    )
                )}
            </strong>

            <span>
                ${clusterEscapeHtml(
                    item.year
                )}
                /
                PAGE
                ${clusterEscapeHtml(
                    item.page
                )}
            </span>

            <span>
                CONFIDENCE
                ${clusterFormatPercent(
                    item.confidence
                )}
            </span>

            <span>
                MEMBERSHIP
                ${clusterFormatPercent(
                    item.cluster_probability
                )}
            </span>
        `;

        tooltip.classList.add(
            "visible"
        );

        positionTooltip(
            event
        );
    }


    function positionTooltip(event) {
        if (!tooltip) {
            return;
        }

        const container =
            map.parentElement;

        const bounds =
            container
                .getBoundingClientRect();

        let left =
            event.clientX
            - bounds.left
            + 14;

        let top =
            event.clientY
            - bounds.top
            + 14;

        const tooltipWidth =
            tooltip.offsetWidth || 250;

        const tooltipHeight =
            tooltip.offsetHeight || 150;

        if (
            left + tooltipWidth
            > bounds.width
        ) {
            left =
                event.clientX
                - bounds.left
                - tooltipWidth
                - 14;
        }

        if (
            top + tooltipHeight
            > bounds.height
        ) {
            top =
                event.clientY
                - bounds.top
                - tooltipHeight
                - 14;
        }

        tooltip.style.left =
            `${Math.max(0, left)}px`;

        tooltip.style.top =
            `${Math.max(0, top)}px`;
    }


    function hideTooltip() {
        if (!tooltip) {
            return;
        }

        tooltip.classList.remove(
            "visible"
        );
    }


    /* ======================================================
       MAP EVENTS
       ====================================================== */

    function addMapEvents(
        mark,
        item
    ) {
        mark.addEventListener(
            "mouseenter",
            event => {
                showTooltip(
                    event,
                    item
                );
            }
        );

        mark.addEventListener(
            "mousemove",
            event => {
                positionTooltip(
                    event
                );
            }
        );

        mark.addEventListener(
            "mouseleave",
            hideTooltip
        );

        mark.addEventListener(
            "click",
            () => {
                openClusterModal(
                    item
                );
            }
        );

        mark.addEventListener(
            "keydown",
            event => {
                if (
                    event.key === "Enter"
                    || event.key === " "
                ) {
                    event.preventDefault();

                    openClusterModal(
                        item
                    );
                }
            }
        );
    }


    /* ======================================================
       MAP RENDERING
       ====================================================== */

    function renderMap() {
        const visibleItems =
            getFilteredItems();

        const currentRenderMode =
            renderMode
                ? renderMode.value
                : "points";

        if (mapStatus) {
            const modeLabel =
                currentRenderMode
                    === "thumbnails"
                    ? "IMAGE THUMBNAILS"
                    : "COLORED POINTS";

            mapStatus.textContent =
                `${visibleItems.length
                    .toLocaleString()} OF `
                + `${allItems.length
                    .toLocaleString()} IMAGES `
                + `/ ${modeLabel}`;
        }

        map.innerHTML = "";

        const width = 1000;
        const height = 650;
        const padding = 38;

        const bounds =
            getBounds(allItems);

        const scaleX = value =>
            padding
            + (
                (
                    value - bounds.minX
                )
                / (
                    bounds.maxX
                    - bounds.minX
                )
            )
            * (
                width
                - padding * 2
            );

        const scaleY = value =>
            height
            - padding
            - (
                (
                    value - bounds.minY
                )
                / (
                    bounds.maxY
                    - bounds.minY
                )
            )
            * (
                height
                - padding * 2
            );

        const namespace =
            "http://www.w3.org/2000/svg";


        /* --------------------------------------------------
           Background
           -------------------------------------------------- */

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
            String(width)
        );

        background.setAttribute(
            "height",
            String(height)
        );

        background.setAttribute(
            "class",
            "cluster-map-background"
        );

        map.appendChild(
            background
        );


        /* --------------------------------------------------
           Grid
           -------------------------------------------------- */

        for (
            let gridIndex = 1;
            gridIndex < 10;
            gridIndex += 1
        ) {
            const vertical =
                document.createElementNS(
                    namespace,
                    "line"
                );

            const horizontal =
                document.createElementNS(
                    namespace,
                    "line"
                );

            const x =
                gridIndex
                * width
                / 10;

            const y =
                gridIndex
                * height
                / 10;

            vertical.setAttribute(
                "x1",
                String(x)
            );

            vertical.setAttribute(
                "x2",
                String(x)
            );

            vertical.setAttribute(
                "y1",
                "0"
            );

            vertical.setAttribute(
                "y2",
                String(height)
            );

            vertical.setAttribute(
                "class",
                "cluster-grid-line"
            );

            horizontal.setAttribute(
                "x1",
                "0"
            );

            horizontal.setAttribute(
                "x2",
                String(width)
            );

            horizontal.setAttribute(
                "y1",
                String(y)
            );

            horizontal.setAttribute(
                "y2",
                String(y)
            );

            horizontal.setAttribute(
                "class",
                "cluster-grid-line"
            );

            map.appendChild(
                vertical
            );

            map.appendChild(
                horizontal
            );
        }


        /* --------------------------------------------------
           Marks
           -------------------------------------------------- */

        const fragment =
            document.createDocumentFragment();

        for (
            const item
            of visibleItems
        ) {
            const xValue =
                Number(item.x);

            const yValue =
                Number(item.y);

            if (
                !Number.isFinite(xValue)
                || !Number.isFinite(yValue)
            ) {
                continue;
            }

            const centerX =
                scaleX(xValue);

            const centerY =
                scaleY(yValue);

            let mark;

            if (
                currentRenderMode
                === "thumbnails"
                && item.crop_image
            ) {
                /*
                 * SVG group containing:
                 *   - colored cluster frame;
                 *   - image thumbnail;
                 *   - transparent interaction rectangle.
                 */
                const group =
                    document.createElementNS(
                        namespace,
                        "g"
                    );

                group.setAttribute(
                    "class",
                    "cluster-thumbnail-mark"
                );

                group.setAttribute(
                    "tabindex",
                    "0"
                );

                group.setAttribute(
                    "role",
                    "button"
                );

                group.setAttribute(
                    "aria-label",
                    `${
                        clusterLabel(
                            item.cluster
                        )
                    }, ${
                        item.year
                        ?? "unknown year"
                    }`
                );


                const frame =
                    document.createElementNS(
                        namespace,
                        "rect"
                    );

                frame.setAttribute(
                    "x",
                    String(
                        centerX - 19
                    )
                );

                frame.setAttribute(
                    "y",
                    String(
                        centerY - 19
                    )
                );

                frame.setAttribute(
                    "width",
                    "38"
                );

                frame.setAttribute(
                    "height",
                    "38"
                );

                frame.setAttribute(
                    "fill",
                    pointColor(item)
                );

                frame.setAttribute(
                    "stroke",
                    "#181818"
                );

                frame.setAttribute(
                    "stroke-width",
                    "1.2"
                );


                const image =
                    document.createElementNS(
                        namespace,
                        "image"
                    );

                image.setAttribute(
                    "x",
                    String(
                        centerX - 15
                    )
                );

                image.setAttribute(
                    "y",
                    String(
                        centerY - 15
                    )
                );

                image.setAttribute(
                    "width",
                    "30"
                );

                image.setAttribute(
                    "height",
                    "30"
                );

                image.setAttribute(
                    "href",
                    item.crop_image
                );

                image.setAttributeNS(
                    "http://www.w3.org/1999/xlink",
                    "href",
                    item.crop_image
                );

                image.setAttribute(
                    "preserveAspectRatio",
                    "xMidYMid slice"
                );


                const hitbox =
                    document.createElementNS(
                        namespace,
                        "rect"
                    );

                hitbox.setAttribute(
                    "x",
                    String(
                        centerX - 20
                    )
                );

                hitbox.setAttribute(
                    "y",
                    String(
                        centerY - 20
                    )
                );

                hitbox.setAttribute(
                    "width",
                    "40"
                );

                hitbox.setAttribute(
                    "height",
                    "40"
                );

                hitbox.setAttribute(
                    "fill",
                    "transparent"
                );

                hitbox.setAttribute(
                    "pointer-events",
                    "all"
                );


                group.appendChild(
                    frame
                );

                group.appendChild(
                    image
                );

                group.appendChild(
                    hitbox
                );

                mark = group;

            } else {
                const point =
                    document.createElementNS(
                        namespace,
                        "circle"
                    );

                point.setAttribute(
                    "cx",
                    String(centerX)
                );

                point.setAttribute(
                    "cy",
                    String(centerY)
                );

                point.setAttribute(
                    "r",
                    Number(
                        item.cluster
                    ) === -1
                        ? "3.5"
                        : "5"
                );

                point.setAttribute(
                    "fill",
                    pointColor(item)
                );

                point.setAttribute(
                    "class",
                    Number(
                        item.cluster
                    ) === -1
                        ? (
                            "cluster-point "
                            + "noise"
                        )
                        : "cluster-point"
                );

                point.setAttribute(
                    "tabindex",
                    "0"
                );

                point.setAttribute(
                    "role",
                    "button"
                );

                point.setAttribute(
                    "aria-label",
                    `${
                        clusterLabel(
                            item.cluster
                        )
                    }, ${
                        item.year
                        ?? "unknown year"
                    }`
                );

                mark = point;
            }

            addMapEvents(
                mark,
                item
            );

            fragment.appendChild(
                mark
            );
        }

        map.appendChild(
            fragment
        );
    }


    /* ======================================================
       CLUSTER ATLAS HELPERS
       ====================================================== */

    function dominantValue(
        items,
        field
    ) {
        const counts =
            new Map();

        for (const item of items) {
            const value =
                item[field]
                ?? "Unknown";

            counts.set(
                value,
                (
                    counts.get(value)
                    || 0
                ) + 1
            );
        }

        let selected =
            "Unknown";

        let maximum = 0;

        for (
            const [value, count]
            of counts.entries()
        ) {
            if (count > maximum) {
                selected = value;
                maximum = count;
            }
        }

        return {
            value: selected,
            count: maximum,
            proportion:
                items.length > 0
                    ? maximum
                        / items.length
                    : 0
        };
    }


    function representativeItems(
        items,
        limit
    ) {
        const sorted =
            [...items];

        const sortByMembership =
            !representativeOnly
            || representativeOnly.checked;

        if (sortByMembership) {
            sorted.sort(
                (a, b) =>
                    Number(
                        b.cluster_probability
                    )
                    - Number(
                        a.cluster_probability
                    )
            );

        } else {
            sorted.sort(
                (a, b) =>
                    Number(
                        b.confidence
                    )
                    - Number(
                        a.confidence
                    )
            );
        }

        if (limit === "all") {
            return sorted;
        }

        return sorted.slice(
            0,
            Number(limit)
        );
    }


    /* ======================================================
       CLUSTER ATLAS
       ====================================================== */

    function renderAtlas() {
        const limit =
            exampleCount
                ? exampleCount.value
                : "12";

        const includeNoise =
            showNoise
                ? showNoise.checked
                : false;

        const groups =
            new Map();

        for (const item of allItems) {
            const clusterId =
                Number(
                    item.cluster
                );

            if (
                clusterId === -1
                && !includeNoise
            ) {
                continue;
            }

            if (
                !groups.has(
                    clusterId
                )
            ) {
                groups.set(
                    clusterId,
                    []
                );
            }

            groups.get(
                clusterId
            ).push(item);
        }

        const orderedGroups = [
            ...groups.entries()
        ].sort(
            (
                [clusterA],
                [clusterB]
            ) => {
                if (clusterA === -1) {
                    return 1;
                }

                if (clusterB === -1) {
                    return -1;
                }

                return (
                    clusterA
                    - clusterB
                );
            }
        );

        atlas.innerHTML = "";

        if (atlasStatus) {
            atlasStatus.textContent =
                `${orderedGroups.length} GROUPS / `
                + `${
                    limit === "all"
                        ? "ALL"
                        : limit
                } EXAMPLES PER GROUP`;
        }

        const fragment =
            document.createDocumentFragment();

        for (
            const [clusterId, items]
            of orderedGroups
        ) {
            const section =
                document.createElement(
                    "section"
                );

            section.className =
                "cluster-atlas-group";

            section.id =
                `cluster-group-${clusterId}`;

            const dominantYear =
                dominantValue(
                    items,
                    "year"
                );

            const meanConfidence =
                items.reduce(
                    (sum, item) =>
                        sum
                        + Number(
                            item.confidence
                        ),
                    0
                )
                / Math.max(
                    items.length,
                    1
                );

            const meanMembership =
                items.reduce(
                    (sum, item) =>
                        sum
                        + Number(
                            item
                                .cluster_probability
                        ),
                    0
                )
                / Math.max(
                    items.length,
                    1
                );

            const examples =
                representativeItems(
                    items,
                    limit
                );

            const clusterColor =
                clusterColors.get(
                    clusterId
                )
                || "#99958c";

            section.innerHTML = `
                <header
                    class="cluster-atlas-header"
                    style="
                        --cluster-color:
                        ${clusterColor};
                    "
                >
                    <div>
                        <h3>
                            ${clusterEscapeHtml(
                                clusterLabel(
                                    clusterId
                                )
                            )}
                        </h3>

                        <p>
                            ${items.length
                                .toLocaleString()}
                            INDEXED IMAGES
                        </p>
                    </div>

                    <div
                        class="cluster-atlas-statistics"
                    >
                        <span>
                            DOMINANT YEAR
                            <strong>
                                ${clusterEscapeHtml(
                                    dominantYear.value
                                )}
                            </strong>
                        </span>

                        <span>
                            MEAN CONFIDENCE
                            <strong>
                                ${clusterFormatPercent(
                                    meanConfidence
                                )}
                            </strong>
                        </span>

                        <span>
                            MEAN MEMBERSHIP
                            <strong>
                                ${clusterFormatPercent(
                                    meanMembership
                                )}
                            </strong>
                        </span>
                    </div>
                </header>

                <div
                    class="cluster-example-grid"
                ></div>
            `;

            const grid =
                section.querySelector(
                    ".cluster-example-grid"
                );

            for (
                const item
                of examples
            ) {
                const card =
                    document.createElement(
                        "button"
                    );

                card.type =
                    "button";

                card.className =
                    "cluster-example-card";

                const image =
                    item.crop_image
                        ? `
                            <img
                                src="${clusterEscapeHtml(
                                    item.crop_image
                                )}"
                                loading="lazy"
                                alt="Visual element from ${
                                    clusterEscapeHtml(
                                        item.year
                                    )
                                }"
                            />
                        `
                        : `
                            <span
                                class="
                                    cluster-missing-image
                                "
                            >
                                IMAGE UNAVAILABLE
                            </span>
                        `;

                card.innerHTML = `
                    <span
                        class="cluster-example-image"
                    >
                        ${image}
                    </span>

                    <span
                        class="cluster-example-details"
                    >
                        <strong>
                            ${clusterEscapeHtml(
                                clusterLabel(
                                    item.cluster
                                )
                            )}
                        </strong>

                        <small>
                            ${clusterEscapeHtml(
                                item.year
                            )}
                            /
                            PAGE
                            ${clusterEscapeHtml(
                                item.page
                            )}
                        </small>

                        <small>
                            MEMBERSHIP
                            ${clusterFormatPercent(
                                item
                                    .cluster_probability
                            )}
                        </small>
                    </span>
                `;

                card.addEventListener(
                    "click",
                    () => {
                        openClusterModal(
                            item
                        );
                    }
                );

                grid.appendChild(
                    card
                );
            }

            fragment.appendChild(
                section
            );
        }

        atlas.appendChild(
            fragment
        );
    }


    /* ======================================================
       MODAL
       ====================================================== */

    function openClusterModal(item) {
        if (
            !modal
            || !modalBody
        ) {
            return;
        }

        const cropImage =
            item.crop_image
                ? `
                    <img
                        class="modal-image"
                        src="${clusterEscapeHtml(
                            item.crop_image
                        )}"
                        alt="Cluster example"
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
                        src="${clusterEscapeHtml(
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

        const confidence =
            clusterFormatPercent(
                item.confidence
            );

        const membership =
            clusterFormatPercent(
                item.cluster_probability
            );

        const outlierScore =
            Number.isFinite(
                Number(
                    item.outlier_score
                )
            )
                ? Number(
                    item.outlier_score
                ).toFixed(3)
                : "N/A";

        modalBody.innerHTML = `
            <div class="modal-grid">

                <section>

                    <h3>
                        Extracted image
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
                    ${clusterEscapeHtml(
                        clusterLabel(
                            item.cluster
                        )
                    )}
                </h3>

                <dl>

                    <dt>Year</dt>
                    <dd>
                        ${clusterEscapeHtml(
                            item.year
                        )}
                    </dd>

                    <dt>Page</dt>
                    <dd>
                        ${clusterEscapeHtml(
                            item.page
                        )}
                    </dd>

                    <dt>Journal</dt>
                    <dd>
                        ${clusterEscapeHtml(
                            item.journal
                        )}
                    </dd>

                    <dt>YOLO confidence</dt>
                    <dd>
                        ${confidence}
                    </dd>

                    <dt>Cluster membership</dt>
                    <dd>
                        ${membership}
                    </dd>

                    <dt>Outlier score</dt>
                    <dd>
                        ${outlierScore}
                    </dd>

                    <dt>Element ID</dt>
                    <dd>
                        <code>
                            ${clusterEscapeHtml(
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
                                href="${clusterEscapeHtml(
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


    /* ======================================================
       EVENT LISTENERS
       ====================================================== */

    if (renderMode) {
        renderMode.addEventListener(
            "change",
            renderMap
        );
    }

    if (colorMode) {
        colorMode.addEventListener(
            "change",
            renderMap
        );
    }

    if (clusterFilter) {
        clusterFilter.addEventListener(
            "change",
            renderMap
        );
    }

    if (classFilter) {
        classFilter.addEventListener(
            "change",
            renderMap
        );
    }

    if (yearFilter) {
        yearFilter.addEventListener(
            "change",
            renderMap
        );
    }

    if (exampleCount) {
        exampleCount.addEventListener(
            "change",
            renderAtlas
        );
    }

    if (showNoise) {
        showNoise.addEventListener(
            "change",
            renderAtlas
        );
    }

    if (representativeOnly) {
        representativeOnly.addEventListener(
            "change",
            renderAtlas
        );
    }

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
                    event.target === modal
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


    /* ======================================================
       INITIAL RENDER
       ====================================================== */

    renderSummary();
    renderMap();
    renderAtlas();
}