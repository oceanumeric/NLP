{
    library(pacman)
    # install.packages("SPARQL_1.16.tar.gz", repos = NULL, type="source")
    p_load(readxl, stringr, data.table, magrittr, ggplot2, fastText, SPARQL,
            eurostat, XML, RCurl, knitr, stringdist, r2d3, RColorBrewer,
            patchwork, cowplot, pheatmap)

    gray_scale <- c('#F3F4F8','#D2D4DA', '#B3B5BD', 
                    '#9496A1', '#7d7f89', '#777986', 
                    '#656673', '#5B5D6B', '#4d505e',
                    '#404352', '#2b2d3b', '#282A3A',
                    '#1b1c2a', '#191a2b',
                    '#141626', '#101223')

    ft_palette <- c('#990F3D', '#0D7680', '#0F5499', '#262A33', '#0b5c3a', '#750db5')

    ft_contrast <- c('#F83', '#00A0DD', '#C00', '#006F9B', '#F2DFCE', '#FF7FAA',
                    '#00994D', '#593380')

    peep_head <- function(dt, n = 5) {
        dt %>%
            head(n) %>%

            kable()
    }

    peep_sample <- function(dt, n = 5) {
        dt %>%
            .[sample(.N, n)] %>%
            kable()
    }

    peep_tail <- function(dt, n = 5) {
        dt %>%
            tail(n) %>%
            kable()
    }
}
# ------------------------------------------------------------------------------

endpoint <- "https://data.epo.org/linked-data/query"

query <- "
prefix patent: <http://data.epo.org/linked-data/def/patent/>
prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

SELECT ?application ?appNum ?filingDate ?authority
WHERE {
?application rdf:type patent:Application ;
    patent:applicationNumber ?appNum ;
    patent:filingDate        ?filingDate ; 
    patent:applicationAuthority ?authority ;
    .
} LIMIT 10
"
qd <- SPARQL(endpoint,query,curl_args=list(useragent=R.version.string))
df <- qd$results

df %>%
    as.data.table() %>%
    peep_head()


# ------------------------------------------------------------------------------
# an applicant's publication -- HUAWEI
query <- "
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#> 
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> 
PREFIX vcard: <http://www.w3.org/2006/vcard/ns#> 
PREFIX patent: <http://data.epo.org/linked-data/def/patent/>

SELECT * 
WHERE {
    ?publn rdf:type patent:Publication;
           patent:applicantVC ?applicant;
           patent:publicationAuthority ?auth;
           patent:publicationNumber ?publnNum;
           patent:publicationKind ?kind;
           patent:publicationDate ?publnDate.
    ?applicant vcard:fn ?name.
    FILTER regex(?name, 'HUAWEI', 'i')
    FILTER (?publnDate >= xsd:date('2019-01-01'))
}
LIMIT 3000
"

hw_query <- SPARQL(endpoint, query, curl_args=list(useragent=R.version.string))
hw_df <- hw_query$results

hw_df %>%
    as.data.table() %>%
    # convert the date to a date object with as.POSIXct
    .[, publnDate := as.POSIXct(publnDate)] %>% str()


# @ai
# need to design query that can return the results within a certain time frame
# otherwise, the query will return 'timeout' error