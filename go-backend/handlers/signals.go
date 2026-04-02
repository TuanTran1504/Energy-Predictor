package handlers

import (
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
)

type SignalsResponse struct {
	BrentCrude BrentResult   `json:"brent_crude"`
	Crypto     CryptoSignals `json:"crypto"`
	FetchedAt  string        `json:"fetched_at"`
}

func Signals(c *gin.Context) {
	brentCh := make(chan *BrentResult, 1)
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
		signals, err := FetchCryptoSignals()
		if err != nil {
			cryptoCh <- &CryptoSignals{}
			return
		}
		cryptoCh <- signals
	}()

	brent := <-brentCh
	crypto := <-cryptoCh

	c.JSON(http.StatusOK, SignalsResponse{
		BrentCrude: *brent,
		Crypto:     *crypto,
		FetchedAt:  time.Now().UTC().Format(time.RFC3339),
	})
}
