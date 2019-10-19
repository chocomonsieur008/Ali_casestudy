construct_reg_formula <- function(data, cols) {
  
  reg.formula <- "class ~"
  
  cols.index <- cols + 1
  
  for (col in cols.index) {
    reg.formula <- paste(reg.formula, "+", colnames(data)[col])
  }
  
  return(as.formula(reg.formula))
}


get_mc_fadden_R2 <- function(logit.model){
  
  loglike.null <- logit.model$null.deviance/-2
  loglike.proposed <- logit.model$deviance/-2
  
  ## McFadden's Pseudo R2 
  mc.fadden.R2 <- (loglike.null - loglike.proposed) / loglike.null
  
  # The p-value for the R2
  mc.fadden.p.value <- 1 - pchisq(2*(loglike.proposed - loglike.null), df = (length(logit.model$coefficients) - 1))
  
  return(list("mc.fadden.R2" = mc.fadden.R2, 
              "mcfadden.p.value" = mc.fadden.p.value))
}


get_concordance_discordance <- function(model) {
  
  fitted <- data.frame(cbind(model$y, model$fitted.values))
  colnames(fitted) <- c('respvar','score')
  
  ones <- fitted[fitted[,1] == 1,]
  zeros <- fitted[fitted[,1] == 0,]
  
  # Initialize all the values
  pairs_tested <- nrow(ones)*nrow(zeros)
  conc <- 0
  disc <- 0
  
  for (i in 1:nrow(ones)) {
    conc <- conc + sum(ones[i,"score"] > zeros[,"score"])
    disc <- disc + sum(ones[i,"score"] < zeros[,"score"])
  }
  
  # Calculate concordance, discordance and ties
  concordance <- conc/pairs_tested
  discordance <- disc/pairs_tested
  ties_perc <- (1 - concordance - discordance)
  
  return(list("Concordance" = concordance,
              "Discordance" = discordance,
              "Tied" = ties_perc,
              "Pairs" = pairs_tested))
}