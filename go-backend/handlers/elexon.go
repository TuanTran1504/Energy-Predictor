package handlers

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// These structs mirror the shape of Elexon's JSON response.
// Go needs to know the exact shape before it can parse JSON —
// unlike Python where you just do response.json()["data"].
type elexonResponse struct {
	Data []elexonDataPoint `json:"data"`
}

type elexonDataPoint struct {
	SettlementPeriod int     `json:"settlementPeriod"`
	SystemSellPrice  float64 `json:"systemSellPrice"`
	SystemBuyPrice   float64 `json:"systemBuyPrice"`
}

// fetchImbalancePrice calls the Elexon API and returns
// the most recent system sell price in £/MWh.
// No API key needed — Elexon is fully public.
func fetchImbalancePrice() (float64, error) {
	yesterday := time.Now().UTC().AddDate(0, 0, -1).Format("2006-01-02")

	url := fmt.Sprintf(
		"https://data.elexon.co.uk/bmrs/api/v1/balancing/settlement/system-prices/%s?format=json",
		yesterday,
	)

	// http.Get is Go's equivalent of httpx.get() in Python.
	// It returns the response AND an error — Go never throws exceptions,
	// it returns errors as values. You must check them explicitly.
	resp, err := http.Get(url)
	if err != nil {
		return 0, fmt.Errorf("elexon request failed: %w", err)
	}
	defer resp.Body.Close() // always close the body when done — Go requires this

	if resp.StatusCode != http.StatusOK {
		return 0, fmt.Errorf("elexon returned status %d", resp.StatusCode)
	}

	// Read the response body into bytes
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return 0, fmt.Errorf("reading elexon response: %w", err)
	}

	// Parse JSON into our struct
	var result elexonResponse
	if err := json.Unmarshal(body, &result); err != nil {
		return 0, fmt.Errorf("parsing elexon json: %w", err)
	}

	if len(result.Data) == 0 {
		return 0, fmt.Errorf("elexon returned no data")
	}

	// Return the most recent period's sell price
	latest := result.Data[len(result.Data)-1]
	return latest.SystemSellPrice, nil
}
