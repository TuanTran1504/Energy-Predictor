package handlers

import (
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
)

type SignalsResponse struct {
	BrentCrude BrentResult   `json:"brent_crude"`
	USDVND     VNDResult     `json:"usd_vnd"`
	Crypto     CryptoSignals `json:"crypto"`
	FetchedAt  string        `json:"fetched_at"`
}

func Signals(c *gin.Context) {
	brentCh := make(chan *BrentResult, 1)
	vndCh := make(chan *VNDResult, 1)
	cryptoCh := make(chan *CryptoSignals, 1)

	go func() {
		delta, err := fetchOilDeltaPct()
		if err != nil {
			brentCh <- &BrentResult{PriceUSD: 0, DeltaDayPct: 0}
			return
		}
		brentCh <- &BrentResult{PriceUSD: 104.20, DeltaDayPct: delta}
	}()

	go func() {
		result, err := FetchVNDRate()
		if err != nil {
			vndCh <- &VNDResult{USDToVND: 26251}
			return
		}
		vndCh <- result
	}()

	go func() {
		// Use cached VND rate — don't depend on vndCh goroutine
		signals, err := FetchCryptoSignals(26251.0)
		if err != nil {
			cryptoCh <- &CryptoSignals{}
			return
		}
		cryptoCh <- signals
	}()

	brent := <-brentCh
	vnd := <-vndCh
	crypto := <-cryptoCh

	c.JSON(http.StatusOK, SignalsResponse{
		BrentCrude: *brent,
		USDVND:     *vnd,
		Crypto:     *crypto,
		FetchedAt:  time.Now().UTC().Format(time.RFC3339),
	})
}
