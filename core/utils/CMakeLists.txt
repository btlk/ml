project(utils)

set(utils_headers
  utils.h
  initializer.h
  data.h
  metric.h
)

set(utils_sources
  src/utils.cpp
  src/initializer.cpp
  src/data.cpp
  src/metric.cpp
)

#set(utils_libs)

add_library(utils STATIC ${utils_sources} ${utils_headers})
#target_link_libraries(utils ${utils_libs})

install(TARGETS utils DESTINATION lib)
install(FILES ${utils_headers} DESTINATION include/utils)

install(TARGETS utils
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)