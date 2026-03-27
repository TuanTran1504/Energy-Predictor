package handlers

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"

	"github.com/gin-gonic/gin"
)

// ForecastRequest is what the dashboard sends to Go
type ForecastRequest struct {
	CurrentLoadMw float64 `json:"current_load_mw"`
	TemperatureC  float64 `json:"temperature_c"`
	WindSpeedMs   float64 `json:"wind_speed_ms"`
	Hour          int     `json:"hour"`
	Weekday       int     `json:"weekday"`
	IsHoliday     bool    `json:"is_holiday"`
	LoadLag1h     float64 `json:"load_lag_1h"`
	LoadLag24h    float64 `json:"load_lag_24h"`
	LoadLag168h   float64 `json:"load_lag_168h"`
}

// PythonPredictRequest is what Go sends to the Python service
// It includes everything ForecastRequest has PLUS the live shock signals
type PythonPredictRequest struct {
	CurrentLoadMw      float64 `json:"current_load_mw"`
	TemperatureC       float64 `json:"temperature_c"`
	WindSpeedMs        float64 `json:"wind_speed_ms"`
	Hour               int     `json:"hour"`
	Weekday            int     `json:"weekday"`
	IsHoliday          bool    `json:"is_holiday"`
	LoadLag1h          float64 `json:"load_lag_1h"`
	LoadLag24h         float64 `json:"load_lag_24h"`
	LoadLag168h        float64 `json:"load_lag_168h"`
	OilPriceBrent      float64 `json:"oil_price_brent"`
	OilDeltaPct30m     float64 `json:"oil_delta_pct_30m"`
	HeadlineScore      float64 `json:"headline_score"`
	GridImbalancePrice float64 `json:"grid_imbalance_price"`
}

// PythonPredictResponse is what the Python service sends back
type PythonPredictResponse struct {
	ForecastMw    float64 `json:"forecast_mw"`
	LowerBoundMw  float64 `json:"lower_bound_mw"`
	UpperBoundMw  float64 `json:"upper_bound_mw"`
	ShockScore    float64 `json:"shock_score"`
	ShockActive   bool    `json:"shock_active"`
	ConfidencePct int     `json:"confidence_pct"`
}

// ForecastResponse is what Go returns to the dashboard
// It combines the Python prediction with the live shock signals
type ForecastResponse struct {
	ForecastMw        float64 `json:"forecast_mw"`
	LowerBoundMw      float64 `json:"lower_bound_mw"`
	UpperBoundMw      float64 `json:"upper_bound_mw"`
	ShockScore        float64 `json:"shock_score"`
	ShockActive       bool    `json:"shock_active"`
	AlertLevel        string  `json:"alert_level"`
	AlertColor        string  `json:"alert_color"`
	OilDeltaPct       float64 `json:"oil_delta_pct"`
	HeadlineScore     float64 `json:"headline_score"`
	ImbalancePriceMwh float64 `json:"imbalance_price_gbp_mwh"`
	Timestamp         string  `json:"timestamp"`
}

func Forecast(c *gin.Context) {
	// 1. Parse the incoming request from the dashboard
	var req ForecastRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid request: " + err.Error()})
		return
	}

	// 2. Fetch all three live signals in parallel using goroutines
	// This is where Go shines — three API calls simultaneously,
	// not one after another like Python asyncio
	type result struct {
		oilDelta  float64
		headline  float64
		imbalance float64
		err       error
	}

	oilCh := make(chan float64, 1)
	newsCh := make(chan float64, 1)
	gridCh := make(chan float64, 1)

	// Launch three goroutines simultaneously
	go func() {
		v, err := fetchOilDeltaPct()
		if err != nil {
			fmt.Printf("Warning: EIA fetch failed: %v\n", err)
			v = 0
		}
		oilCh <- v
	}()

	go func() {
		v, err := fetchHeadlineScore()
		if err != nil {
			fmt.Printf("Warning: GNews fetch failed: %v\n", err)
			v = 0
		}
		newsCh <- v
	}()

	go func() {
		v, err := fetchImbalancePrice()
		if err != nil {
			fmt.Printf("Warning: Elexon fetch failed: %v\n", err)
			v = 0
		}
		gridCh <- v
	}()

	// Wait for all three to finish
	oilDelta := <-oilCh
	headlineScore := <-newsCh
	imbalancePrice := <-gridCh

	fmt.Printf("Signals fetched — oil: %.2f%% headline: %.2f imbalance: £%.2f\n",
		oilDelta, headlineScore, imbalancePrice)

	// 3. Build the request for the Python ML service
	pythonReq := PythonPredictRequest{
		CurrentLoadMw:      req.CurrentLoadMw,
		TemperatureC:       req.TemperatureC,
		WindSpeedMs:        req.WindSpeedMs,
		Hour:               req.Hour,
		Weekday:            req.Weekday,
		IsHoliday:          req.IsHoliday,
		LoadLag1h:          req.LoadLag1h,
		LoadLag24h:         req.LoadLag24h,
		LoadLag168h:        req.LoadLag168h,
		OilPriceBrent:      104.0, // placeholder — add EIA price endpoint later
		OilDeltaPct30m:     oilDelta,
		HeadlineScore:      headlineScore,
		GridImbalancePrice: imbalancePrice,
	}

	// 4. Call the Python ML service
	prediction, err := callPythonMLService(pythonReq)
	if err != nil {
		// If Python service is down, return a degraded response
		// rather than crashing — this is called graceful degradation
		fmt.Printf("Warning: Python ML service unavailable: %v\n", err)
		c.JSON(http.StatusServiceUnavailable, gin.H{
			"error":  "ML service unavailable",
			"detail": "Model not trained yet or Python service is down",
		})
		return
	}

	// 5. Build and return the final response
	_, alertColor, _ := getAlertLevel(prediction.ShockScore)
	level, _, alertMsg := getAlertLevel(prediction.ShockScore)

	c.JSON(http.StatusOK, ForecastResponse{
		ForecastMw:        prediction.ForecastMw,
		LowerBoundMw:      prediction.LowerBoundMw,
		UpperBoundMw:      prediction.UpperBoundMw,
		ShockScore:        prediction.ShockScore,
		ShockActive:       prediction.ShockActive,
		AlertLevel:        level,
		AlertColor:        alertColor,
		OilDeltaPct:       oilDelta,
		HeadlineScore:     headlineScore,
		ImbalancePriceMwh: imbalancePrice,
		Timestamp:         time.Now().UTC().Format(time.RFC3339),
	})

	_ = alertMsg // used in dashboard — suppress unused warning
}

// callPythonMLService sends the feature vector to the Python inference
// service and returns the prediction. Pure HTTP — no shared memory,
// no tight coupling. Go and Python stay completely independent.
func callPythonMLService(req PythonPredictRequest) (*PythonPredictResponse, error) {
	pythonURL := os.Getenv("PYTHON_ML_URL")
	if pythonURL == "" {
		pythonURL = "http://localhost:8000" // default Python FastAPI port
	}

	// Marshal the request to JSON
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshaling request: %w", err)
	}
	client := &http.Client{Timeout: 5 * time.Second}
	// POST to Python /predict endpoint
	resp, err := client.Post(
		pythonURL+"/api/forecast",
		"application/json",
		bytes.NewBuffer(body),
	)
	if err != nil {
		return nil, fmt.Errorf("calling Python service: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == 503 {
		return nil, fmt.Errorf("Python service returned 503 — model not loaded yet")
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("Python service returned status %d", resp.StatusCode)
	}

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("reading Python response: %w", err)
	}

	var prediction PythonPredictResponse
	if err := json.Unmarshal(respBody, &prediction); err != nil {
		return nil, fmt.Errorf("parsing Python response: %w", err)
	}

	return &prediction, nil
}
