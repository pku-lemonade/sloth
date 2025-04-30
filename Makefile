.PHONY: run test clean

B ?= 0
E ?= 20000000000

# 下面这些变量可以在命令行传入，例如：make run FLOW=--flow
FLOW ?=


run:
	python run.py --simstart=$(B) --simend=$(E) $(FLOW)

test:
	python test.py

clean:
	rm -rf gen
	rm -rf build
