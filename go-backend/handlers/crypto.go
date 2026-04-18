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
	SOL       CryptoPrice `json:"sol"`
	XRP       CryptoPrice `json:"xrp"`
	FetchedAt string      `json:"fetched_at"`
}

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

	displaySymbol := symbol[:len(symbol)-4]

	return &CryptoPrice{
		Symbol:       displaySymbol,
		PriceUSD:     price,
		Change24hPct: changePct,
		Volume24hUSD: volume,
		LastUpdated:  time.Now().UTC().Format(time.RFC3339),
	}, nil
}

func FetchCryptoSignals() (*CryptoSignals, error) {
	btcCh := make(chan *CryptoPrice, 1)
	ethCh := make(chan *CryptoPrice, 1)
	solCh := make(chan *CryptoPrice, 1)
	xrpCh := make(chan *CryptoPrice, 1)

	for _, pair := range []struct {
		sym string
		ch  chan *CryptoPrice
	}{
		{"BTCUSDT", btcCh},
		{"ETHUSDT", ethCh},
		{"SOLUSDT", solCh},
		{"XRPUSDT", xrpCh},
	} {
		pair := pair
		go func() {
			price, err := fetchCryptoPrice(pair.sym)
			if err != nil {
				fmt.Printf("Warning: %s fetch failed: %v\n", pair.sym, err)
				pair.ch <- &CryptoPrice{Symbol: pair.sym[:len(pair.sym)-4], PriceUSD: 0}
				return
			}
			pair.ch <- price
		}()
	}

	btc := <-btcCh
	eth := <-ethCh
	sol := <-solCh
	xrp := <-xrpCh

	return &CryptoSignals{
		BTC:       *btc,
		ETH:       *eth,
		SOL:       *sol,
		XRP:       *xrp,
		FetchedAt: time.Now().UTC().Format(time.RFC3339),
	}, nil
}

// ── Candle history via Binance klines ─────────────────────────────────────────

type CryptoCandle struct {
	Timestamp string  `json:"timestamp"`
	Open      float64 `json:"open"`
	High      float64 `json:"high"`
	Low       float64 `json:"low"`
	Close     float64 `json:"close"`
	Volume    float64 `json:"volume"`
	ChangePct float64 `json:"change_pct"`
}

type CryptoHistoryResponse struct {
	Symbol   string         `json:"symbol"`
	Interval string         `json:"interval"`
	Candles  []CryptoCandle `json:"candles"`
	Count    int            `json:"count"`
}

// intervalToBinance maps our interval strings to Binance kline intervals
func intervalToBinance(interval string) string {
	switch interval {
	case "30m":
		return "30m"
	case "1H":
		return "1h"
	case "4H":
		return "4h"
	case "1D":
		return "1d"
	case "1W":
		return "1w"
	case "1M":
		return "1M"
	default:
		return "1h"
	}
}

// candlesForDays returns how many candles cover the requested day range
// for a given interval, capped at Binance's max of 1000
func candlesForDays(interval string, days int) int {
	var candlesPerDay float64
	switch interval {
	case "30m":
		candlesPerDay = 48
	case "1H":
		candlesPerDay = 24
	case "4H":
		candlesPerDay = 6
	case "1D":
		candlesPerDay = 1
	case "1W":
		candlesPerDay = 1.0 / 7
	case "1M":
		candlesPerDay = 1.0 / 30
	default:
		candlesPerDay = 24
	}
	limit := int(float64(days) * candlesPerDay)
	if limit < 1 {
		limit = 1
	}
	if limit > 1000 {
		limit = 1000
	}
	return limit
}

func fetchBinanceKlines(symbol, interval string, days int) ([]CryptoCandle, error) {
	binanceInterval := intervalToBinance(interval)
	limit := candlesForDays(interval, days)

	url := fmt.Sprintf(
		"%s/klines?symbol=%sUSDT&interval=%s&limit=%d",
		binanceBaseURL, symbol, binanceInterval, limit,
	)

	client := &http.Client{Timeout: 15 * time.Second}
	resp, err := client.Get(url)
	if err != nil {
		return nil, fmt.Errorf("binance klines request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("reading klines response: %w", err)
	}

	// Binance klines: array of arrays
	// [openTime, open, high, low, close, volume, closeTime, ...]
	var raw [][]json.RawMessage
	if err := json.Unmarshal(body, &raw); err != nil {
		return nil, fmt.Errorf("parsing klines response: %w", err)
	}

	parseStr := func(r json.RawMessage) float64 {
		var s string
		if err := json.Unmarshal(r, &s); err != nil {
			return 0
		}
		f, _ := strconv.ParseFloat(s, 64)
		return f
	}

	parseInt64 := func(r json.RawMessage) int64 {
		var n int64
		json.Unmarshal(r, &n)
		return n
	}

	candles := make([]CryptoCandle, 0, len(raw))
	var prevClose float64

	for _, k := range raw {
		if len(k) < 6 {
			continue
		}

		openTimeMs := parseInt64(k[0])
		t := time.UnixMilli(openTimeMs).UTC()
		open := parseStr(k[1])
		high := parseStr(k[2])
		low := parseStr(k[3])
		close_ := parseStr(k[4])
		volume := parseStr(k[5])

		changePct := 0.0
		if prevClose > 0 {
			changePct = (close_ - prevClose) / prevClose * 100
		}
		prevClose = close_

		candles = append(candles, CryptoCandle{
			Timestamp: t.Format(time.RFC3339),
			Open:      open,
			High:      high,
			Low:       low,
			Close:     close_,
			Volume:    volume,
			ChangePct: changePct,
		})
	}

	return candles, nil
}

// ── HTTP handlers ─────────────────────────────────────────────────────────────

func CryptoStatus(c *gin.Context) {
	signals, err := FetchCryptoSignals()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, signals)
}

// fetchFuturesMarkPrice fetches the mark price from Binance Futures API.
// Mark price is used for uPnL calculation, unlike spot last price.
func fetchFuturesMarkPrice(symbol string) (float64, error) {
	url := fmt.Sprintf("https://fapi.binance.com/fapi/v1/premiumIndex?symbol=%s", symbol)
	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Get(url)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return 0, err
	}
	var data struct {
		MarkPrice string `json:"markPrice"`
	}
	if err := json.Unmarshal(body, &data); err != nil {
		return 0, err
	}
	return strconv.ParseFloat(data.MarkPrice, 64)
}

// CryptoPrices handles GET /crypto/prices — returns live futures mark prices for BTC/ETH/SOL.
// Used for polling uPnL in the frontend every few seconds.
func CryptoPrices(c *gin.Context) {
	symbols := []string{"BTCUSDT", "ETHUSDT", "SOLUSDT"}
	type result struct {
		sym   string
		price float64
	}
	ch := make(chan result, len(symbols))
	for _, sym := range symbols {
		sym := sym
		go func() {
			price, err := fetchFuturesMarkPrice(sym)
			if err != nil {
				price = 0
			}
			ch <- result{sym[:len(sym)-4], price}
		}()
	}
	prices := make(map[string]float64, len(symbols))
	for range symbols {
		r := <-ch
		prices[r.sym] = r.price
	}
	c.JSON(http.StatusOK, prices)
}

// CryptoHistory handles GET /crypto/history?symbol=BTC&interval=1H&days=30
// Fetches directly from Binance klines — no DB dependency.
func CryptoHistory(c *gin.Context) {
	symbol := c.DefaultQuery("symbol", "BTC")
	interval := c.DefaultQuery("interval", "1H")
	daysStr := c.DefaultQuery("days", "30")

	days, err := strconv.Atoi(daysStr)
	if err != nil || days < 1 || days > 365 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "days must be between 1 and 365"})
		return
	}

	// Validate interval
	validIntervals := map[string]bool{
		"30m": true, "1H": true, "4H": true,
		"1D": true, "1W": true, "1M": true,
	}
	if !validIntervals[interval] {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "invalid interval. Use: 30m, 1H, 4H, 1D, 1W, 1M",
		})
		return
	}

	candles, err := fetchBinanceKlines(symbol, interval, days)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, CryptoHistoryResponse{
		Symbol:   symbol,
		Interval: interval,
		Candles:  candles,
		Count:    len(candles),
	})
}
