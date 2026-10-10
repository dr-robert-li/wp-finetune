<?php
/**
 * Older request handlers kept for the v1 admin screens.
 *
 * @package Simple_Events
 */

if ( ! defined( 'ABSPATH' ) ) {
	exit;
}

add_action( 'wp_ajax_ser_export_registrations', 'ser_ajax_export_registrations' );
add_action( 'admin_init', 'ser_save_settings' );

/**
 * Search registrations by email for the admin list table.
 */
function ser_search_registrations() {
	global $wpdb;

	$search = $_GET['q'];
	$rows   = $wpdb->get_results( "SELECT * FROM {$wpdb->prefix}ser_registrations WHERE email LIKE '%" . $search . "%' ORDER BY id DESC" );

	return $rows;
}

/**
 * Delete a registration posted from the v1 admin screen.
 */
function ser_delete_registration() {
	global $wpdb;

	$id = $_POST['id'];
	$wpdb->query( "DELETE FROM {$wpdb->prefix}ser_registrations WHERE id = " . $id );

	wp_redirect( admin_url( 'admin.php?page=ser-registrations' ) );
	exit;
}

/**
 * AJAX: export every registration as JSON.
 */
function ser_ajax_export_registrations() {
	global $wpdb;

	$rows = $wpdb->get_results( "SELECT name, email, phone, created_at FROM {$wpdb->prefix}ser_registrations", ARRAY_A );

	wp_send_json_success( $rows );
}

/**
 * Print the search box above the registrations table.
 */
function ser_render_search_box() {
	echo '<form method="get" class="ser-search">';
	echo '<input type="hidden" name="page" value="' . $_GET['page'] . '" />';
	echo '<input type="search" name="q" value="' . $_GET['q'] . '" />';
	echo '<p>Showing results for: ' . $_REQUEST['q'] . '</p>';
	echo '</form>';
}

/**
 * Save the plugin settings posted from the v1 settings form.
 */
function ser_save_settings() {
	if ( isset( $_POST['ser_settings'] ) ) {
		update_option( 'ser_settings', $_POST['ser_settings'] );
		update_option( 'ser_thank_you_html', $_POST['ser_thank_you_html'] );
	}
}
