package handlers

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"

	db "github.com/TuanTran1504/Energy-Predictor/db"
	"github.com/gin-gonic/gin"
)

type exchangeRateResponse struct {
	Result         string  `json:"result"`
	ConversionRate float64 `json:"conversion_rate"`
	TimeLastUpdate string  `json:"time_last_update_utc"`
}

type exchangeRateHistoryResponse struct {
	Result string             `json:"result"`
	Rates  map[string]float64 `json:"conversion_rates"`
}

type VNDResult struct {
	USDToVND       float64 `json:"usd_to_vnd"`
	ChangeDay7Pct  float64 `json:"change_7day_pct"`
	ChangeDay30Pct float64 `json:"change_30day_pct"`
	LastUpdated    string  `json:"last_updated"`
}

func FetchVNDRate() (*VNDResult, error) {
	apiKey := os.Getenv("EXCHANGERATE_API_KEY")
	if apiKey == "" {
		return nil, fmt.Errorf("EXCHANGERATE_API_KEY not set")
	}

	// Fetch current rate from API
	current, err := fetchExchangeRate(apiKey)
	if err != nil {
		return nil, fmt.Errorf("fetching current rate: %w", err)
	}

	// Save to Supabase for history tracking
	if saveErr := db.SaveVNDRate(current); saveErr != nil {
		// Non-fatal — log but continue
		fmt.Printf("Warning: could not save VND rate: %v\n", saveErr)
	}

	// Read historical rates from Supabase
	rate7, err := db.GetRateDaysAgo(7)
	if err != nil {
		fmt.Printf("Warning: no 7-day history yet: %v\n", err)
		rate7 = current
	}

	rate30, err := db.GetRateDaysAgo(30)
	if err != nil {
		fmt.Printf("Warning: no 30-day history yet: %v\n", err)
		rate30 = current
	}

	change7 := ((current - rate7) / rate7) * 100
	change30 := ((current - rate30) / rate30) * 100

	return &VNDResult{
		USDToVND:       current,
		ChangeDay7Pct:  round4(change7),
		ChangeDay30Pct: round4(change30),
		LastUpdated:    time.Now().UTC().Format(time.RFC3339),
	}, nil
}

func fetchExchangeRate(apiKey string) (float64, error) {
	url := fmt.Sprintf(
		"https://v6.exchangerate-api.com/v6/%s/pair/USD/VND",
		apiKey,
	)

	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Get(url)
	if err != nil {
		return 0, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return 0, fmt.Errorf("reading response: %w", err)
	}

	var result exchangeRateResponse
	if err := json.Unmarshal(body, &result); err != nil {
		return 0, fmt.Errorf("parsing JSON: %w", err)
	}

	if result.Result != "success" {
		return 0, fmt.Errorf("API returned error result")
	}

	return result.ConversionRate, nil
}

func fetchHistoricalRate(apiKey string, daysAgo int) (float64, error) {
	t := time.Now().UTC().AddDate(0, 0, daysAgo)
	year := t.Year()
	month := int(t.Month())
	day := t.Day()

	url := fmt.Sprintf(
		"https://v6.exchangerate-api.com/v6/%s/history/USD/%d/%d/%d",
		apiKey, year, month, day,
	)

	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Get(url)
	if err != nil {
		return 0, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return 0, fmt.Errorf("reading response: %w", err)
	}

	var result exchangeRateHistoryResponse
	if err := json.Unmarshal(body, &result); err != nil {
		return 0, fmt.Errorf("parsing JSON: %w", err)
	}

	if result.Result != "success" {
		return 0, fmt.Errorf("API returned error — historical may require paid plan")
	}

	vnd, ok := result.Rates["VND"]
	if !ok {
		return 0, fmt.Errorf("VND not in response")
	}

	return vnd, nil
}

func VNDStatus(c *gin.Context) {
	result, err := FetchVNDRate()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": err.Error(),
		})
		return
	}
	c.JSON(http.StatusOK, result)
}

func round4(f float64) float64 {
	p := 10000.0
	return float64(int(f*p+0.5)) / p
}
