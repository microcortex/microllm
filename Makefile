# MicroCortex project helpers

.PHONY: help new-recipe lint check

NEXT_ID := $(shell ls recipes | grep -E '^[0-9]{3}-' | sort -r | head -1 | cut -d- -f1 | awk '{printf "%03d", $$1+1}')

help:
	@echo "MicroCortex — tiny LLMs born for \$5 chips"
	@echo ""
	@echo "  make new-recipe NAME=my-recipe   Create recipes/$(NEXT_ID)-my-recipe from TEMPLATE"
	@echo "  make lint                        Lint all Markdown"
	@echo "  make check                       Verify repo structure (recipes must have required sections)"

new-recipe:
	@if [ -z "$(NAME)" ]; then echo "Usage: make new-recipe NAME=my-recipe"; exit 1; fi
	mkdir -p recipes/$(NEXT_ID)-$(NAME)
	cp recipes/TEMPLATE.md recipes/$(NEXT_ID)-$(NAME)/README.md
	@echo "✅ Created recipes/$(NEXT_ID)-$(NAME)/ — go fill it in!"

lint:
	npx --yes markdownlint-cli2 "**/*.md" "#node_modules"

check:
	@bash -c 'fail=0; for f in recipes/0*/README.md; do \
		for section in "Hardware" "Quickstart" "Results" "Reproduce"; do \
			grep -q "$$section" "$$f" || { echo "❌ $$f missing section: $$section"; fail=1; }; \
		done; \
	done; [ $$fail -eq 0 ] && echo "✅ All recipes have required sections"'
