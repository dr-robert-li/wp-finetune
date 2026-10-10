<?php
// Third-party helper bundled by composer. Must be skipped (vendor).
function vendor_unsafe_query( $id ) {
	global $wpdb;
	return $wpdb->get_results( "SELECT * FROM wp_posts WHERE ID = " . $_GET['id'] );
}
