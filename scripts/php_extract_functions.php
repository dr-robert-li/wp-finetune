<?php
/**
 * PHP Function Extractor for WordPress Finetuning Pipeline.
 *
 * Extracts individual functions, methods, and class definitions from a PHP file
 * along with their PHPDoc blocks and dependency references.
 *
 * Usage: php php_extract_functions.php <file_path>
 * Output: JSON array to stdout
 */

if ($argc < 2) {
    fwrite(STDERR, "Usage: php php_extract_functions.php <file_path>\n");
    exit(1);
}

$file_path = $argv[1];

if (!file_exists($file_path)) {
    fwrite(STDERR, "File not found: {$file_path}\n");
    exit(1);
}

$source = file_get_contents($file_path);
$tokens = token_get_all($source);
$functions = [];

$current_docblock = null;
$brace_depth = 0;
$in_function = false;
$in_class = false;
$current_class = null;
$function_start_line = 0;
$function_tokens = [];
$function_name = '';
$function_brace_start = 0;

for ($i = 0; $i < count($tokens); $i++) {
    $token = $tokens[$i];

    if (is_array($token)) {
        list($token_type, $token_value, $token_line) = $token;

        // Capture doc comments.
        if ($token_type === T_DOC_COMMENT) {
            $current_docblock = $token_value;
            continue;
        }

        // Track class context.
        if ($token_type === T_CLASS) {
            // Look ahead for class name.
            for ($j = $i + 1; $j < count($tokens); $j++) {
                if (is_array($tokens[$j]) && $tokens[$j][0] === T_STRING) {
                    $current_class = $tokens[$j][1];
                    $in_class = true;
                    break;
                }
            }
        }

        // Detect function/method declarations.
        if ($token_type === T_FUNCTION && !$in_function) {
            // Look ahead for function name.
            $fname = null;
            for ($j = $i + 1; $j < count($tokens); $j++) {
                if (is_array($tokens[$j]) && $tokens[$j][0] === T_STRING) {
                    $fname = $tokens[$j][1];
                    break;
                }
                // Anonymous function — skip.
                if (!is_array($tokens[$j]) && $tokens[$j] === '(') {
                    break;
                }
            }

            if ($fname !== null) {
                $in_function = true;
                $function_name = $fname;
                $function_start_line = $token_line;
                $function_tokens = [];
                $function_brace_start = $brace_depth;
            }
        }
    }

    // Track braces.
    if (!is_array($token)) {
        if ($token === '{') {
            $brace_depth++;
        } elseif ($token === '}') {
            $brace_depth--;

            // Function ended.
            if ($in_function && $brace_depth === $function_brace_start) {
                $body = '';
                foreach ($function_tokens as $ft) {
                    $body .= is_array($ft) ? $ft[1] : $ft;
                }
                $body .= '}';

                // Extract dependency references.
                $dependencies = extract_dependencies($body);

                // Extract SQL usage.
                $sql_patterns = extract_sql_patterns($body);

                // Extract hook usage.
                $hooks = extract_hooks($body);

                $full_name = $current_class ? "{$current_class}::{$function_name}" : $function_name;

                $functions[] = [
                    'function_name'  => $full_name,
                    'class_context'  => $current_class,
                    'docblock'       => $current_docblock,
                    'body'           => $body,
                    'start_line'     => $function_start_line,
                    'dependencies'   => $dependencies,
                    'sql_patterns'   => $sql_patterns,
                    'hooks_used'     => $hooks,
                    'line_count'     => substr_count($body, "\n") + 1,
                ];

                $in_function = false;
                $current_docblock = null;
                $function_name = '';
                continue;
            }

            // Class ended.
            if ($in_class && $brace_depth === 0) {
                $in_class = false;
                $current_class = null;
            }
        }
    }

    // Collect function body tokens.
    if ($in_function) {
        $function_tokens[] = $token;
    }

    // Clear docblock if next meaningful token isn't a function/class.
    if (is_array($token) && $token[0] !== T_WHITESPACE && $token[0] !== T_DOC_COMMENT
        && $token[0] !== T_FUNCTION && $token[0] !== T_CLASS && $token[0] !== T_ABSTRACT
        && $token[0] !== T_PUBLIC && $token[0] !== T_PROTECTED && $token[0] !== T_PRIVATE
        && $token[0] !== T_STATIC && $token[0] !== T_FINAL) {
        if (!$in_function) {
            $current_docblock = null;
        }
    }
}

echo json_encode($functions, JSON_PRETTY_PRINT);

// ─── Helper functions ────────────────────────────────

function extract_dependencies(string $body): array {
    $deps = [];
    // Match function calls that look like custom WP functions (not built-in PHP).
    if (preg_match_all('/\b([a-z_][a-z0-9_]*)\s*\(/i', $body, $matches)) {
        $php_builtins = ['isset', 'empty', 'unset', 'array', 'list', 'echo', 'print',
            'die', 'exit', 'return', 'if', 'else', 'elseif', 'foreach', 'for', 'while',
            'switch', 'case', 'break', 'continue', 'function', 'class', 'new', 'try',
            'catch', 'throw', 'finally', 'count', 'strlen', 'strpos', 'substr',
            'array_map', 'array_filter', 'array_merge', 'array_keys', 'array_values',
            'in_array', 'is_array', 'is_string', 'is_int', 'is_null', 'intval',
            'absint', 'sprintf', 'printf', 'implode', 'explode', 'trim', 'strtolower',
            'strtoupper', 'json_encode', 'json_decode', 'preg_match', 'preg_replace',
            'str_replace', 'array_push', 'array_pop', 'array_shift', 'array_unique',
            'sort', 'usort', 'ksort', 'compact', 'extract', 'defined', 'define',
            'max', 'min', 'ceil', 'floor', 'round', 'time', 'date', 'strtotime',
        ];
        foreach (array_unique($matches[1]) as $func) {
            if (!in_array(strtolower($func), $php_builtins, true)) {
                $deps[] = $func;
            }
        }
    }
    // Match static method calls.
    if (preg_match_all('/([A-Z][a-zA-Z0-9_]*)::\s*([a-z_][a-z0-9_]*)\s*\(/i', $body, $matches)) {
        for ($i = 0; $i < count($matches[0]); $i++) {
            $deps[] = $matches[1][$i] . '::' . $matches[2][$i];
        }
    }
    return array_values(array_unique($deps));
}

function extract_sql_patterns(string $body): array {
    $patterns = [];
    if (strpos($body, '$wpdb') !== false) {
        if (strpos($body, '->prepare(') !== false) {
            $patterns[] = 'prepared_query';
        }
        if (preg_match('/->get_results\s*\(/', $body)) {
            $patterns[] = 'get_results';
        }
        if (preg_match('/->get_var\s*\(/', $body)) {
            $patterns[] = 'get_var';
        }
        if (preg_match('/->get_row\s*\(/', $body)) {
            $patterns[] = 'get_row';
        }
        if (preg_match('/->get_col\s*\(/', $body)) {
            $patterns[] = 'get_col';
        }
        if (preg_match('/->query\s*\(/', $body)) {
            $patterns[] = 'direct_query';
        }
        if (preg_match('/->insert\s*\(/', $body)) {
            $patterns[] = 'insert';
        }
        if (preg_match('/->update\s*\(/', $body)) {
            $patterns[] = 'update';
        }
        if (preg_match('/->delete\s*\(/', $body)) {
            $patterns[] = 'delete';
        }
        if (stripos($body, 'JOIN') !== false) {
            $patterns[] = 'join';
        }
        if (stripos($body, 'dbDelta') !== false || stripos($body, 'dbdelta') !== false) {
            $patterns[] = 'dbdelta';
        }
    }
    return $patterns;
}

function extract_hooks(string $body): array {
    $hooks = [];
    if (preg_match_all('/(add_action|add_filter|do_action|apply_filters|remove_action|remove_filter)\s*\(\s*[\'"]([^\'"]+)[\'"]/i', $body, $matches)) {
        for ($i = 0; $i < count($matches[0]); $i++) {
            $hooks[] = $matches[1][$i] . '(' . $matches[2][$i] . ')';
        }
    }
    return $hooks;
}
