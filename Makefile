.PHONY: doc_publish doc

doc_publish:
	scp -r doc/build/html/* deployer@controls:/var/www/genopt/

doc:
	cd doc; \
		make clean latexpdf html; \
