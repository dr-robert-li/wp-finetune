<?php
// bad fixture 2 - style violations, poor practices, no docblocks

if (!defined('ABSPATH')) exit;

class bad_class_02 {{
    var $Value;
    var $Data;

    function __construct($v) {{
        $this->Value = $v;
        $this->Data = array();
    }}

    function getStuff() {{
        return $this->Value;
    }}

    function setStuff($v) {{
        $this->Value = $v;
    }}

    function do_thing($x,$y,$z) {{
        $result = $x+$y+$z;
        return $result;
    }}

    function processData( $d ) {{
        foreach($d as $k=>$v) {{
            $this->Data[$k] = $v;
        }}
        return $this->Data;
    }}

    function renderOutput() {{
        $val = esc_html( $this->Value );
        echo '<p>' . $val . '</p>';
    }}

    function getCount() {{
        return count($this->Data);
    }}

    function maybe_run($flag) {{
        if($flag == true) {{
            return $this->do_thing(1,2,3);
        }}
        return null;
    }}
}}

function bad_helper_02($input) {{
    $result = array();
    if (is_array($input)) {{
        foreach($input as $item) {{
            $result[] = sanitize_text_field($item);
        }}
    }}
    return $result;
}}

function bad_runner_02() {{
    $obj = new bad_class_02('test');
    $data = bad_helper_02(array('a','b','c'));
    $obj->processData($data);
    return $obj->getCount();
}}
