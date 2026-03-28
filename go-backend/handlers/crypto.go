package handlers

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"time"

	"github.com/gin-gonic/gin"
)

type CryptoPrice struct {
	Symbol       string  `json:"symbol"`
	PriceUSD     float64 `json:"price_usd"`
	Change24hPct float64 `json:"change_24h_pct"`
	Volume24hUSD float64 `json:"volume_24h_usd"`
	LastUpdated  string  `json:"last_updated"`
}

type CryptoSignals struct {
	BTC       CryptoPrice `json:"btc"`
	ETH       CryptoPrice `json:"eth"`
	USDTToVND float64     `json:"usdt_to_vnd"`
	FetchedAt string      `json:"fetched_at"`
}

// Binance public API — no key needed for market data
const binanceBaseURL = "https://api.binance.com/api/v3"

type binanceTicker struct {
	Symbol         string `json:"symbol"`
	PriceChange    string `json:"priceChange"`
	PriceChangePct string `json:"priceChangePercent"`
	LastPrice      string `json:"lastPrice"`
	QuoteVolume    string `json:"quoteVolume"`
}

func fetchCryptoPrice(symbol string) (*CryptoPrice, error) {
	url := fmt.Sprintf("%s/ticker/24hr?symbol=%s", binanceBaseURL, symbol)

	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Get(url)
	if err != nil {
		return nil, fmt.Errorf("binance request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("reading response: %w", err)
	}

	var ticker binanceTicker
	if err := json.Unmarshal(body, &ticker); err != nil {
		return nil, fmt.Errorf("parsing response: %w", err)
	}

	price, _ := strconv.ParseFloat(ticker.LastPrice, 64)
	changePct, _ := strconv.ParseFloat(ticker.PriceChangePct, 64)
	volume, _ := strconv.ParseFloat(ticker.QuoteVolume, 64)

	// Strip USDT suffix for display
	displaySymbol := symbol[:len(symbol)-4]

	return &CryptoPrice{
		Symbol:       displaySymbol,
		PriceUSD:     price,
		Change24hPct: changePct,
		Volume24hUSD: volume,
		LastUpdated:  time.Now().UTC().Format(time.RFC3339),
	}, nil
}

func FetchCryptoSignals(vndRate float64) (*CryptoSignals, error) {
	// Fetch BTC and ETH simultaneously
	btcCh := make(chan *CryptoPrice, 1)
	ethCh := make(chan *CryptoPrice, 1)

	go func() {
		price, err := fetchCryptoPrice("BTCUSDT")
		if err != nil {
			fmt.Printf("Warning: BTC fetch failed: %v\n", err)
			btcCh <- &CryptoPrice{Symbol: "BTC", PriceUSD: 0}
			return
		}
		btcCh <- price
	}()

	go func() {
		price, err := fetchCryptoPrice("ETHUSDT")
		if err != nil {
			fmt.Printf("Warning: ETH fetch failed: %v\n", err)
			ethCh <- &CryptoPrice{Symbol: "ETH", PriceUSD: 0}
			return
		}
		ethCh <- price
	}()

	btc := <-btcCh
	eth := <-ethCh

	return &CryptoSignals{
		BTC:       *btc,
		ETH:       *eth,
		USDTToVND: vndRate, // 1 USDT ≈ 1 USD in VND terms
		FetchedAt: time.Now().UTC().Format(time.RFC3339),
	}, nil
}

func CryptoStatus(c *gin.Context) {
	// Get current VND rate for USDT/VND display
	vndResult, err := FetchVNDRate()
	vndRate := 26251.0
	if err == nil {
		vndRate = vndResult.USDToVND
	}

	signals, err := FetchCryptoSignals(vndRate)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": err.Error(),
		})
		return
	}

	c.JSON(http.StatusOK, signals)
}
