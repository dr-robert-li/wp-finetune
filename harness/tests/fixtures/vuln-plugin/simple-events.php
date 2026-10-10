<?php
/**
 * Plugin Name:       Simple Event Registrations
 * Plugin URI:        https://example.com/simple-events
 * Description:       Lets visitors register for events and lets admins manage the registrations.
 * Version:           1.2.0
 * Requires at least: 6.0
 * Requires PHP:      7.4
 * Author:            Example Co
 * License:           GPL-2.0-or-later
 * Text Domain:       simple-events
 */

if ( ! defined( 'ABSPATH' ) ) {
	exit;
}

require_once __DIR__ . '/includes/legacy.php';

add_action( 'admin_post_ser_cancel_registration', 'ser_handle_cancel_registration' );
add_filter( 'sanitize_option_ser_settings', 'ser_sanitize_settings' );

/**
 * Text domain used by every translatable string in the plugin.
 */
function ser_text_domain() { return 'simple-events'; }

/**
 * Fetch the registrations for one event.
 *
 * @param int $event_id Event post ID.
 * @return array Registration rows.
 */
function ser_get_registrations_for_event( $event_id ) {
	global $wpdb;

	$event_id = absint( $event_id );
	if ( ! $event_id ) {
		return array();
	}

	$table = $wpdb->prefix . 'ser_registrations';

	return $wpdb->get_results(
		$wpdb->prepare(
			"SELECT id, name, email, created_at FROM {$table} WHERE event_id = %d ORDER BY created_at DESC",
			$event_id
		),
		ARRAY_A
	);
}

/**
 * Print the number of registrations for an event.
 *
 * @param int $event_id Event post ID.
 */
function ser_render_registration_count( $event_id ) {
	$count = count( ser_get_registrations_for_event( $event_id ) );

	printf(
		'<p class="ser-count">%s</p>',
		esc_html(
			sprintf(
				/* translators: %d: number of registrations. */
				_n( '%d registration', '%d registrations', $count, 'simple-events' ),
				$count
			)
		)
	);
}

/**
 * Cancel a registration from the admin screen (admin-post handler).
 */
function ser_handle_cancel_registration() {
	if ( ! current_user_can( 'manage_options' ) ) {
		wp_die( esc_html__( 'You are not allowed to cancel registrations.', 'simple-events' ), 403 );
	}

	check_admin_referer( 'ser_cancel_registration' );

	$registration_id = isset( $_POST['registration_id'] ) ? absint( wp_unslash( $_POST['registration_id'] ) ) : 0;
	if ( $registration_id ) {
		global $wpdb;
		$wpdb->delete( $wpdb->prefix . 'ser_registrations', array( 'id' => $registration_id ), array( '%d' ) );
	}

	wp_safe_redirect( admin_url( 'admin.php?page=ser-registrations&cancelled=1' ) );
	exit;
}

/**
 * Sanitize the settings array before it is stored.
 *
 * @param array $input Raw settings from the options form.
 * @return array Clean settings.
 */
function ser_sanitize_settings( $input ) {
	$input = is_array( $input ) ? $input : array();

	return array(
		'notify_email' => isset( $input['notify_email'] ) ? sanitize_email( $input['notify_email'] ) : '',
		'capacity'     => isset( $input['capacity'] ) ? absint( $input['capacity'] ) : 0,
		'thank_you'    => isset( $input['thank_you'] ) ? sanitize_text_field( $input['thank_you'] ) : '',
	);
}
