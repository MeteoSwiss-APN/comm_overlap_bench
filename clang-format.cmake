# Clang Format CMake target
# Source: https://arcanis.me/en/2015/10/17/cppcheck-and-clang-format
file(GLOB_RECURSE ALL_SOURCE_FILES *.cpp *.h)
foreach (SOURCE_FILE ${ALL_SOURCE_FILES}) 
	string(FIND ${SOURCE_FILE} ${PROJECT_SOURCE_DIR}/external PROJECT_EXTERNAL_DIR_FOUND) 
	if (NOT ${PROJECT_EXTERNAL_DIR_FOUND} EQUAL -1) 
		list(REMOVE_ITEM ALL_SOURCE_FILES ${SOURCE_FILE}) 
	endif () 
endforeach () 

find_program(CLANGFORMAT clang-format
	HINTS /usr/bin /usr/local/bin /usr/local/opt/llvm/bin
	DOC "Clang Format executable")

add_custom_target( clangformat COMMAND ${CLANGFORMAT} -style=file -i ${ALL_SOURCE_FILES} )
