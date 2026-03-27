package main

import (
	"log"
	"os"

	"github.com/gin-gonic/gin"
	"github.com/joho/godotenv"

	"github.com/TuanTran1504/Energy-Predictor/handlers"
)

func main() {
	godotenv.Load("../.env")

	r := gin.Default()

	// Existing route
	r.GET("/health", handleHealth)

	// New route — maps to handlers/shock.go
	r.GET("/shock/status", handlers.ShockStatus)
	r.POST("/forecast", handlers.Forecast)

	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	log.Printf("Go server starting on port %s", port)
	r.Run(":" + port)
}

func handleHealth(c *gin.Context) {
	c.JSON(200, gin.H{
		"status":  "ok",
		"service": "energy-forecaster-go",
	})
}
