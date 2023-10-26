remove_na <- function(x, y) {
	remove_idx <- which(is.na(x[, y]))
	if (length(remove_idx) > 0) {
		x <- x[-remove_idx, ]
	}
	return(x)
}

create_folds <- function(x, endpoints) {
	# create the various folds
	for (colx in endpoints) {
		folds <- createFolds(x[, colx], k=5, list=FALSE)
		x[paste(colx,"_folds", sep="")] <- folds
	}
	return(x)
}

