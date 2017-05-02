project(gene)

set(gene_headers
  gene.h
)

set(gene_sources
  src/gene.cpp
)

set(gene_libs
  utils
)

add_library(gene STATIC ${gene_sources} ${gene_headers})
target_link_libraries(gene ${gene_libs})

if (BUILD_GO)
  add_executable(gene_go src/gene_go.cpp)
  target_link_libraries(gene_go gene)
endif()

install(TARGETS gene DESTINATION lib)
install(FILES ${gene_headers} DESTINATION include/gene)

install(TARGETS gene
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)