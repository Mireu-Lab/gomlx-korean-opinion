//go:build ((linux && (amd64 || arm64)) || (darwin && arm64) || (windows && amd64)) && !noxla

package model

import (
	_ "github.com/gomlx/go-xla/compute/xla/autoinstall"
)
