# Build all three decks. Each deck has its own build.sh.
.PHONY: all clean deep_dive tutorial overview

all: deep_dive tutorial overview

deep_dive:
	bash deep_dive/build.sh

tutorial:
	bash tutorial/build.sh

overview:
	bash overview/build.sh

clean:
	rm -f */*.aux */*.log */*.nav */*.snm */*.toc */*.out */*.synctex.gz
	@echo "cleaned beamer build artifacts (kept .pdf and .tex)"
