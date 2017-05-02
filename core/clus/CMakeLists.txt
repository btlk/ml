project(clus)

set(clus_headers
  clus.h
  forel.h
)

set(clus_sources
  src/clus.cpp
  src/forel.cpp
)

set(clus_libs
  utils
)

add_library(clus STATIC ${clus_sources} ${clus_headers})
target_link_libraries(clus ${clus_libs})

if (BUILD_GO)
  add_executable(forel_go src/forel_go.cpp)
  target_link_libraries(forel_go clus)
endif()

install(TARGETS clus DESTINATION lib)
install(FILES ${clus_headers} DESTINATION include/clus)

install(TARGETS clus
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)