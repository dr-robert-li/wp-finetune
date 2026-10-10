<?php
/**
 * Settings screen markup. Loaded by the admin menu callback.
 *
 * @package Simple_Events
 */

if ( ! defined( 'ABSPATH' ) ) {
	exit;
}

$ser_settings = get_option( 'ser_settings', array() );
?>
<div class="wrap">
	<h1><?php echo esc_html( get_admin_page_title() ); ?></h1>
	<form method="post" action="options.php">
		<?php settings_fields( 'ser_settings' ); ?>
		<table class="form-table" role="presentation">
			<tr>
				<th scope="row"><label for="ser-notify-email"><?php esc_html_e( 'Notification email', 'simple-events' ); ?></label></th>
				<td><input type="email" id="ser-notify-email" name="ser_settings[notify_email]" value="<?php echo esc_attr( $ser_settings['notify_email'] ?? '' ); ?>" class="regular-text" /></td>
			</tr>
			<tr>
				<th scope="row"><label for="ser-capacity"><?php esc_html_e( 'Capacity', 'simple-events' ); ?></label></th>
				<td><input type="number" id="ser-capacity" name="ser_settings[capacity]" value="<?php echo esc_attr( $ser_settings['capacity'] ?? 0 ); ?>" min="0" /></td>
			</tr>
		</table>
		<?php submit_button(); ?>
	</form>
</div>
