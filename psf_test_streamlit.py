# def perform_epsf_photometry_streamlit(image_data, background_data, phot_table, fwhm, daofind, detection_mask):
#     """
#     Wrapper for PSF photometry with Streamlit UI integration.
    
#     Parameters
#     ----------
#     image_data : numpy.ndarray
#         Science image data
#     background_data : numpy.ndarray
#         Background data to subtract
#     phot_table : astropy.table.Table
#         Table with source positions
#     fwhm : float
#         FWHM estimate in pixels
#     daofind : DAOStarFinder
#         Configured DAOStarFinder object
#     detection_mask : int
#         Border size to mask
        
#     Returns
#     -------
#     tuple
#         (phot_epsf_result, epsf_model)
#     """
#     st.write("Starting PSF photometry...")
    
#     # Create mask
#     mask = make_border_mask(image_data, border=detection_mask)
    
#     # Subtract background if provided
#     image_bg_subtracted = image_data - background_data if background_data is not None else image_data
    
#     try:
#         # Perform PSF photometry
#         phot_epsf_result, epsf_model = perform_epsf_photometry(
#             image_bg_subtracted, phot_table, fwhm, daofind, mask
#         )
        
#         # Store results in session state
#         st.session_state['epsf_photometry_result'] = phot_epsf_result
#         st.session_state['epsf_model'] = epsf_model
        
#         st.success("PSF photometry completed successfully.")
        
#         # Display EPSF model
#         st.subheader("PSF Model")
#         norm_epsf = ImageNormalize(epsf_model.data, interval=ZScaleInterval())
#         fig_epsf, ax_epsf = plt.subplots(figsize=FIGURE_SIZES['medium'], dpi=100)
#         im_epsf = ax_epsf.imshow(epsf_model.data, norm=norm_epsf, origin='lower', cmap='viridis')
#         fig_epsf.colorbar(im_epsf, ax=ax_epsf, label='PSF Model Value')
#         ax_epsf.set_title("PSF Model (ZScale)")
#         st.pyplot(fig_epsf)
        
#         # Display photometry results preview
#         st.subheader("PSF Photometry Results (first 10 rows)")
#         preview_df = phot_epsf_result[:10].to_pandas()
#         st.dataframe(preview_df)
        
#         return phot_epsf_result, epsf_model
#     except Exception as e:
#         st.error(f"Error performing PSF photometry: {e}")
#         return None, None