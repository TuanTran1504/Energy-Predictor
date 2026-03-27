package handlers

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
)

type eiaResponse struct {
	Response struct {
		Data []struct {
			Period string `json:"period"`
			Value  string `json:"value"` // EIA returns price as a string
		} `json:"data"`
	} `json:"response"`
}

// fetchOilDeltaPct returns the day-over-day % change in Brent crude.
// This is the heaviest signal in the shock scorer (50% weight).
func fetchOilDeltaPct() (float64, error) {
	apiKey := os.Getenv("EIA_API_KEY")
	if apiKey == "" {
		return 0, fmt.Errorf("EIA_API_KEY not set")
	}

	url := fmt.Sprintf(
		"https://api.eia.gov/v2/petroleum/pri/spt/data/?api_key=%s&frequency=daily&data[0]=value&facets[product][]=EPCBRENT&sort[0][column]=period&sort[0][direction]=desc&length=2&out=json",
		apiKey,
	)

	resp, err := http.Get(url)
	if err != nil {
		return 0, fmt.Errorf("EIA request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return 0, fmt.Errorf("reading EIA response: %w", err)
	}

	var result eiaResponse
	if err := json.Unmarshal(body, &result); err != nil {
		return 0, fmt.Errorf("parsing EIA json: %w", err)
	}

	data := result.Response.Data
	if len(data) < 2 {
		return 0, fmt.Errorf("not enough EIA data points")
	}

	// data[0] = most recent day, data[1] = day before
	today, err := strconv.ParseFloat(data[0].Value, 64)
	if err != nil {
		return 0, fmt.Errorf("parsing today price: %w", err)
	}

	yesterday, err := strconv.ParseFloat(data[1].Value, 64)
	if err != nil {
		return 0, fmt.Errorf("parsing yesterday price: %w", err)
	}

	if yesterday == 0 {
		return 0, fmt.Errorf("yesterday price is zero")
	}

	deltaPct := ((today - yesterday) / yesterday) * 100
	return deltaPct, nil
}
