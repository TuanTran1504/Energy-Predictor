package handlers

import (
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
)

// Ron95Price holds the current and historical Ron 95 gasoline price
// Hardcoded for now — replaced by scraper in next step
type Ron95Price struct {
	PriceVND       float64 `json:"price_vnd"`
	LastAdjusted   string  `json:"last_adjusted"`
	NextAdjustment string  `json:"next_adjustment"`
	DaysUntilNext  int     `json:"days_until_next"`
	Source         string  `json:"source"`
}

// SignalsResponse is what GET /signals returns
// The single source of truth for the dashboard
type SignalsResponse struct {
	BrentCrude BrentResult `json:"brent_crude"`
	USDVND     VNDResult   `json:"usd_vnd"`
	Ron95      Ron95Price  `json:"ron95"`
	FetchedAt  string      `json:"fetched_at"`
}

// Signals handles GET /signals
// Fetches all three Vietnam energy signals in parallel
func Signals(c *gin.Context) {
	// Launch all three fetches simultaneously using goroutines
	brentCh := make(chan *BrentResult, 1)
	vndCh := make(chan *VNDResult, 1)

	go func() {
		delta, err := fetchOilDeltaPct()
		if err != nil {
			brentCh <- &BrentResult{
				PriceUSD:    0,
				DeltaDayPct: 0,
				Error:       err.Error(),
			}
			return
		}
		brentCh <- &BrentResult{
			PriceUSD:    104.20, // placeholder — add price endpoint to eia.go
			DeltaDayPct: delta,
		}
	}()

	go func() {
		result, err := FetchVNDRate()
		if err != nil {
			vndCh <- &VNDResult{
				USDToVND: 0,
			}
			return
		}
		vndCh <- result
	}()

	brent := <-brentCh
	vnd := <-vndCh

	// Ron 95 — hardcoded for now
	// Next step: replace with scraper
	nextAdjustment := "2026-03-27"
	daysUntil := daysUntilDate(nextAdjustment)

	ron95 := Ron95Price{
		PriceVND:       21536,
		LastAdjusted:   "2026-03-17",
		NextAdjustment: nextAdjustment,
		DaysUntilNext:  daysUntil,
		Source:         "Ministry of Industry and Trade (hardcoded — scraper pending)",
	}

	c.JSON(http.StatusOK, SignalsResponse{
		BrentCrude: *brent,
		USDVND:     *vnd,
		Ron95:      ron95,
		FetchedAt:  time.Now().UTC().Format(time.RFC3339),
	})
}

// daysUntilDate returns how many days until a target date string
func daysUntilDate(dateStr string) int {
	target, err := time.Parse("2006-01-02", dateStr)
	if err != nil {
		return 0
	}
	now := time.Now().UTC().Truncate(24 * time.Hour)
	days := int(target.Sub(now).Hours() / 24)
	if days < 0 {
		return 0
	}
	return days
}
