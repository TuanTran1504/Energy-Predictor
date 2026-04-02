package handlers

import (
	"fmt"
	"math"
	"net/http"

	"github.com/gin-gonic/gin"
)

// ShockInput is the data the endpoint expects.
// In Go, structs define the shape of data — like a Pydantic model in Python.
type ShockInput struct {
	OilDeltaPct    float64 `json:"oil_delta_pct"`
	HeadlineScore  float64 `json:"headline_score"`
}

// ShockResponse is what the endpoint returns.
type ShockResponse struct {
	ShockScore  float64 `json:"shock_score"`
	AlertLevel  string  `json:"alert_level"`
	AlertColor  string  `json:"alert_color"`
	Message     string  `json:"message"`
	ShockActive bool    `json:"shock_active"`

	// Raw signals — so you can see exactly what each source contributed
	OilDeltaPct   float64 `json:"oil_delta_pct"`
	HeadlineScore float64 `json:"headline_score"`
}

// ComputeShockScore is the same logic as your Python shock_scorer.py
// translated into Go. Same weights, same thresholds.
func computeShockScore(oilDelta, headlineScore float64) float64 {
	// Only upward oil moves signal supply shock
	oilSignal := math.Min(math.Max(oilDelta, 0)/15.0, 1.0)
	newsSignal := math.Min(math.Max(headlineScore, 0), 1.0)

	score := 0.70*oilSignal + 0.30*newsSignal

	// Round to 4 decimal places
	return math.Round(score*10000) / 10000
}

func getAlertLevel(score float64) (level, color, message string) {
	switch {
	case score >= 0.6:
		return "HIGH", "red", "Geopolitical shock active — forecast uncertainty elevated"
	case score >= 0.3:
		return "ELEVATED", "amber", "Elevated geopolitical risk — monitoring closely"
	default:
		return "NORMAL", "green", "No significant disruption detected"
	}
}

// ShockStatus handles GET /shock/status
// c *gin.Context is like FastAPI's Request — it holds the request and response.
func ShockStatus(c *gin.Context) {
	// Fetch real oil delta from EIA
	oilDelta, err := fetchOilDeltaPct()
	if err != nil {
		fmt.Printf("Warning: EIA fetch failed: %v\n", err)
		oilDelta = 0
	}

	headlineScore, err := fetchHeadlineScore()
	if err != nil {
		fmt.Printf("Warning: NewsAPI fetch failed: %v\n", err)
		headlineScore = 0
	}

	score := computeShockScore(oilDelta, headlineScore)
	level, color, message := getAlertLevel(score)

	c.JSON(http.StatusOK, ShockResponse{
		ShockScore:     score,
		AlertLevel:     level,
		AlertColor:     color,
		Message:        message,
		ShockActive:    score >= 0.6,
		OilDeltaPct:    oilDelta,
		HeadlineScore:  headlineScore,
	})
}
