.PHONY:run test showlink showlsu
B?=0
E?=20000000000
run:
	python run.py --simstart=$(B) --simend=$(E)
test:
	python test.py

showlink:
	python run.py --simstart=$(B) --simend=$(E) --showlink

showtask:
	python run.py --simstart=$(B) --simend=$(E) --showtask