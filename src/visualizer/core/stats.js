export class StatsData {
    constructor(data) {
        this._data = data
    }

    getSentenceScores() {
        return this._data.sentences.map(s => s.lint_score)
    }

    getWordFrequencies() {
        return this._data.sentences
            .flatMap(s => s.word_features)
            .map(wf => wf.word_frequency)
            .filter(freq => freq !== null)
    }

    getContentWordsPerClause() {
        return this._data.sentences.map(s => s.content_words_per_clause)
    }

    getNounCountsByType() {
        const counts = { concrete: 0, abstract: 0, undefined: 0, unknown: 0 }

        this._data.sentences.forEach(s => {
            s.word_features.forEach(wf => {
                if (wf.super_sem_type) {
                    counts[wf.super_sem_type]++
                }
            })
        })

        return counts
    }

    getDependencyLengths() {
        return this._data.sentences
            .flatMap(s => s.word_features)
            .map(wf => wf.dep_length)
            .filter(dl => dl !== undefined && dl > 0)
    }
}

export class StatsSpecs {
    static createScoreBoxPlot(sentScores) {
        return {
            title: "Zinsscore",
            data: { values: sentScores.map(value => ({ value })) },
            mark: { type: "boxplot", extent: "min-max" },
            encoding: {
                y: {
                    field: "value",
                    type: "quantitative",
                    title: "Leesbaarheid",
                    scale: { domain: [0, 100] }
                }
            },
            width: 80
        }
    }

    static createFrequencyBoxPlot(wordFreqs) {
        return {
            title: "Woordfrequentie",
            data: { values: wordFreqs.map(value => ({ value })) },
            mark: { type: "boxplot", extent: "min-max" },
            encoding: {
                y: {
                    field: "value",
                    type: "quantitative",
                    title: "Zipf-frequentie",
                    scale: { zero: false }
                }
            },
            width: 80
        }
    }

    static createContentWordsPerClauseBoxPlot(values) {
        return {
            title: "Inhoudswoorden per deelzin",
            data: { values: values.map(value => ({ value })) },
            mark: { type: "boxplot", extent: "min-max" },
            encoding: {
                y: {
                    field: "value",
                    type: "quantitative",
                    title: "Woorden/deelzin",
                    scale: { zero: false }
                }
            },
            width: 80
        }
    }

    static createNounTypesBarChart(nounCounts, colors) {
        const labelMap = { concrete: 'concreet', abstract: 'abstract', undefined: 'onbepaald', unknown: 'onbekend' }
        const data = Object.entries(nounCounts).map(([type, count]) => ({
            type,
            label: labelMap[type] || type,
            count
        }))

        return {
            title: "Typen zelfstandige naamwoorden",
            data: { values: data },
            mark: "bar",
            encoding: {
                x: {
                    field: "label",
                    type: "nominal",
                    title: null,
                    axis: { labelAngle: 0 },
                    sort: ["concreet", "abstract", "onbepaald", "onbekend"]
                },
                y: {
                    field: "count",
                    type: "quantitative",
                    title: "Aantal"
                },
                color: {
                    field: "type",
                    type: "nominal",
                    scale: {
                        domain: ["concrete", "abstract", "undefined", "unknown"],
                        range: [
                            colors.concrete,
                            colors.abstract,
                            colors.undefined,
                            colors.unknown
                        ]
                    },
                    legend: null
                },
                tooltip: [
                    { field: "label", type: "nominal", title: "Type" },
                    { field: "count", type: "quantitative", title: "Aantal" }
                ]
            },
            width: 250
        }
    }

    static createDependencyLengthHistogram(depLengths) {
        const counts = depLengths.reduce((acc, dl) => {
            acc[dl] = (acc[dl] || 0) + 1
            return acc
        }, {})

        const data = Object.entries(counts).map(([length, count]) => ({
            length: parseInt(length),
            count
        }))

        return {
            title: "Afhankelijkheidslengte",
            data: { values: data },
            mark: "bar",
            encoding: {
                x: {
                    field: "length",
                    type: "ordinal",
                    title: "SDL"
                },
                y: {
                    field: "count",
                    type: "quantitative",
                    title: "Aantal"
                },
                tooltip: [
                    { field: "count", type: "quantitative", title: "Aantal" }
                ]
            },
            width: 200
        }
    }

    static createStatsVisualization({wordFreqs, sentScores, nounCounts, depLengths, contentWordsPerClause}, colors) {
        return {
            $schema: "https://vega.github.io/schema/vega-lite/v5.json",
            vconcat: [
                {
                    hconcat: [
                        this.createScoreBoxPlot(sentScores),
                        this.createFrequencyBoxPlot(wordFreqs),
                        this.createContentWordsPerClauseBoxPlot(contentWordsPerClause),
                    ]
                },
                {
                    hconcat: [
                        this.createNounTypesBarChart(nounCounts, colors),
                        this.createDependencyLengthHistogram(depLengths)
                    ]
                }
            ],
            config: {
                view: { stroke: null },
                background: null,  // transparent background
                axis: {
                    domainColor: colors.currentColor,
                    tickColor: colors.currentColor,
                    gridColor: null,
                    labelColor: colors.currentColor,
                    titleColor: colors.currentColor
                },
                title: {
                    color: colors.currentColor
                },
                rule: {
                    color: colors.currentColor
                }
            }
        }
    }
}
